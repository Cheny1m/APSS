import os
import time
from tqdm import tqdm
import math
import random
import json

import mindspore.dataset as ds
import mindspore.ops as ops
import mindspore as ms
from mindspore import save_checkpoint
from mindspore import Tensor
import mindspore.common.dtype as mstype
from mindspore.communication.management import init

from apss.nets.attention_model import set_decode_type
from apss.utils.log_utils import log_values

from .test import test
from .test import get_partiton_cost_sequence

from apss.utils.reinforce_loss import CustomReinforceLoss

with open('config.json', 'r') as f:
    config = json.load(f)
RESOURCE_DIR = config["RESOURCE_DIR"]

def pi2partition(pi,node_size):
    pi.sort()
    assert node_size > pi[-1]+1, print(node_size,pi)
    piadd1 = [i+1 for i in pi]
    piadd1 = [0] + piadd1 + [node_size]
    partition = []
    for i, p in enumerate(piadd1):
        if i ==0:
            continue
        partition.append(p - piadd1[i-1])
    return partition

# MindSpore's DataParallel mode provides direct access to the model's network.
def get_inner_model(model):
    # parallel_mode = context.get_auto_parallel_context("parallel_mode")
    # if parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL):
    #     return model._network
    # else:
    #     return model
    return model

def validate(model, dataset, opts):
    print('Validating...')
    cost,pi = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, cost.std() / math.sqrt(len(cost))))
    return avg_cost

def rollout(model, dataset, opts):
    set_decode_type(model, "greedy")
    model.set_train(False)

    def eval_model_bat(bat, ori_bat, cost_c_bat):
        cost, _, pi = model(Tensor(bat, ms.float32),
                            Tensor(ori_bat, ms.float32),
                            Tensor(cost_c_bat, ms.float32),
                            return_pi = True)
        return cost,pi
    bats = []
    pis = []
    ms_dataset = ds.GeneratorDataset(source=dataset,column_names=["data", "ori_data", "cost_c_data"],num_parallel_workers=4)
    ms_dataset = ms_dataset.batch(batch_size=opts.eval_batch_size) 
    for data in tqdm(ms_dataset.create_dict_iterator(),total = math.ceil(len(dataset) / opts.eval_batch_size)):
        cost,pi = eval_model_bat(data["data"],data["ori_data"],data["cost_c_data"])
        bats.append(cost)
        pis.append(pi)
    bats = ops.concat(bats, 0)
    pis = ops.concat(pis, 0)
    return bats, pis.asnumpy()

def clip_grad_norms(param_groups, max_norm=math.inf):
    grad_norms = ops.clip_by_global_norm(param_groups,max_norm if max_norm > 0 else math.inf)
    return grad_norms,grad_norms

def train_epoch(model, optimizer, baseline, lr_scheduler,epoch, val_dataset, problem, tb_logger, opts):

    print("Start train epoch {}, lr={} for run {}".format(epoch, lr_scheduler(epoch), opts.run_name))
    
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.group_lr[0](epoch).asnumpy(), step)
    # init()
    # Generate new training data for each epoch
    training_dataset_ = baseline.wrap_dataset(problem.make_dataset(
        size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution, num_split=opts.num_split))
    
    # training_dataset = (data, ori_data, cost_c_data), baseline_cost, baseline_pi
    def data_generator(training_dataset_):
        for item in training_dataset_:
            (data, ori_data, cost_c_data), baseline_cost, baseline_pi = item
            yield data, ori_data, cost_c_data, baseline_cost, baseline_pi

    # Unpacking of data tuples to Tensor/numpy
    training_dataset = ds.GeneratorDataset(source=data_generator(training_dataset_), column_names=['data','ori_data','cost_c_data','baseline_cost', 'baseline_pi'],num_parallel_workers=4)

    training_dataset = training_dataset.batch(batch_size=opts.batch_size,drop_remainder=True)

    # Put model in train mode!
    model.set_train()
    set_decode_type(model, "sampling")
    print("train batch baseline name: ", baseline)#," alpha : ", baseline.alpha)

    for batch_id, batch in enumerate(tqdm(training_dataset.create_dict_iterator(), total = math.ceil(len(training_dataset_) / opts.batch_size),disable=opts.no_progress_bar)):
        train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts
        )
        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        append_dict = {
            'rng_state': ms.get_seed()
            # 'baseline': baseline.state_dict()
        }
        save_checkpoint(optimizer, os.path.join(RESOURCE_DIR,opts.save_dir, 'epoch-{}.ckpt'.format(epoch)),append_dict = append_dict)
    avg_reward = validate(model, val_dataset, opts)

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_reward, step)

    baseline.epoch_callback(model, epoch)

def train_batch(model, optimizer, baseline, epoch, batch_id, step, batch, tb_logger, opts):
    batch = {
        'data': (batch['data'], batch['ori_data'], batch['cost_c_data']),
        'baseline_cost': batch['baseline_cost'],
        'baseline_pi': batch['baseline_pi']
    }
    # print("combined batch",batch,type(batch))
    x, bl_val, bl_pi = baseline.unwrap_batch(batch)
    split_x, ori_x, cost_c_x = x
    # ctx = ms.context.set_context(mode=context.GRAPH_MODE, device_target=opts.device)
    split_x = Tensor(split_x, dtype=mstype.float32)
    ori_x = Tensor(ori_x, dtype=mstype.float32)
    cost_c_x = Tensor(cost_c_x, dtype=mstype.float32)
    bl_val = Tensor(bl_val, dtype=mstype.float32) if bl_val is not None else None

    bl_val, bl_loss = baseline.eval(split_x, ori_x, cost_c_x) if bl_val is None else (bl_val, 0)
    bl_cost = bl_val

    loss_fn = CustomReinforceLoss()

    def forward_fn(split_x, ori_x, cost_c_x, bl_cost):
        cost, log_likelihood, pi = model(split_x, ori_x, cost_c_x, return_pi=True)
        cost2, log_likelihood2, pi2 = model(split_x, ori_x, cost_c_x, return_pi=True)
        loss = loss_fn(cost, cost2, bl_cost, log_likelihood, log_likelihood2)
        return loss,cost,pi,log_likelihood

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    (loss,cost,pi,log_likelihood),grads = grad_fn(split_x, ori_x, cost_c_x, bl_cost)
    # print("type",type(grads),"grads:",grads)
    (grad_norms,grad_norms_clipped) = clip_grad_norms(grads,opts.max_grad_norm)
    optimizer(grad_norms)

    grad_norms1 = (grad_norms,grad_norms_clipped)

    # 日志记录
    if step % opts.log_step == 0:
        # print(f"loss: {reinforce_loss}, {bl_loss}")
        print(f"loss: {loss_fn.get_loss()[0]}, {loss_fn.get_loss()[1]}")
        indexs = random.sample([i for i in range(1)],1)
        for idx in indexs:
            if bl_pi is not None:
                bl_partition = pi2partition(bl_pi[idx, ...].asnumpy().tolist(), model.node_size)   
            partition = pi2partition(pi[idx, ...].asnumpy().tolist(), model.node_size)
            print(f"rl, dp cost: {cost.reshape(-1)[idx]}, bl cost: {bl_cost.reshape(-1)[idx]}, eval bl cost: {get_partiton_cost_sequence(ori_x[idx,...].reshape(-1), cost_c_x[idx,...], bl_partition)}")
            print(f"pi: {partition}")
            if bl_pi is not None:
                print(f"baseline pi: {bl_partition}")
        
        # log_values(cost, grad_norms, epoch, batch_id, step, log_likelihood, reinforce_loss, bl_loss, tb_logger, opts)  
        log_values(cost, grad_norms1, epoch, batch_id, step, log_likelihood, loss_fn.get_loss()[0], loss_fn.get_loss()[1], tb_logger, opts)  

        if opts.run_test:
            set_decode_type(model, "greedy")
            test(model,model.node_size,model.num_split)
            set_decode_type(model, "sampling")
            model.set_train(mode=True)  
    return loss