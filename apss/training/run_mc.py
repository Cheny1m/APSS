import os
import json
import pprint as pp
from tensorboard_logger import Logger as TbLogger

import mindspore as ms
import mindspore.nn as nn
import mindspore.communication as communication
from mindspore.communication import init,get_rank,get_group_size
from mindspore.amp import auto_mixed_precision

from apss.nets.attention_model import AttentionModel
from apss.nets.attention_model_v2 import AttentionStepModel
from apss.utils import load_problem
from apss.utils.reinforce_loss import CustomReinforceLoss

from .options import get_options
from .train_mc import validate,train_all #,train_epoch
from .reinforce_baselines_pp import  RolloutBaselinePP,WarmupBaseline,NoBaseline

import gc
import os
import time
from tqdm import tqdm
import math
import random
import json
import psutil
import sys

import mindspore.dataset as ds
import mindspore.ops as ops
import mindspore as ms
from mindspore import save_checkpoint
from mindspore import Tensor
import mindspore.common.dtype as mstype
from mindspore.communication.management import init

from apss.nets.attention_model import set_decode_type
from apss.utils.log_utils import log_values
# from apss.problems.pp.problem_pp import get_pp_costs

from .test import test
from .test import get_partiton_cost_sequence

from apss.utils.reinforce_loss import CustomReinforceLoss

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

# @profile
def rollout(model, dataset, opts):
    set_decode_type(model, "greedy")
    model.set_train(False)

    def eval_model_bat(bat, ori_bat, cost_c_bat):
        cost, _, pi = model(bat,ori_bat,cost_c_bat,return_pi = True) # 内存增加
        return cost,pi

    bats = []
    pis = []

    ms_dataset = ds.GeneratorDataset(source=dataset,column_names=["data", "ori_data", "cost_c_data"],num_parallel_workers=1)
    ms_dataset = ms_dataset.batch(batch_size=opts.eval_batch_size) 

    for bat, ori_bat,cost_c_bat in tqdm(ms_dataset.create_tuple_iterator(),total = math.ceil(len(dataset) / opts.eval_batch_size)):# # 内存增长
        cost, pi = eval_model_bat(bat,ori_bat,cost_c_bat)
        bats.append(cost)
        pis.append(pi)

    bats = ops.concat(bats, 0)
    pis = ops.concat(pis, 0)
    return bats, pis.asnumpy()

def clip_grad_norms(param_groups, max_norm=math.inf):
    grad_norms = ops.clip_by_global_norm(param_groups,max_norm if max_norm > 0 else math.inf)
    return grad_norms,grad_norms

with open('config.json', 'r') as f:
    config = json.load(f)
RESOURCE_DIR = config["RESOURCE_DIR"]
CONTEXT_MODE = config["CONTEXT_MODE"]
DEVICE_TARGET = config["DEVICE_TARGET"]
    

with open('config.json', 'r') as f:
    config = json.load(f)
RESOURCE_DIR = config["RESOURCE_DIR"]


def run(opts):

    # Pretty print the run args
    print("The run args is:")
    pp.pprint(vars(opts))

    # Set the random seed
    ms.set_seed(opts.seed)

    print("device:",ms.get_context("device_target"),"\nmode:",ms.get_context("mode"))

    # Optionally configure tensorboard/ install tensorflow and tensorboard_logger. mindinsight can be uesed for this.
    # log dir:crlsf-pp/logs
    tb_logger = None
    if not opts.no_tensorboard:
        log_dir = os.path.join(RESOURCE_DIR,opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name)
        tb_logger = TbLogger(log_dir)
        print("Greate TbLogger to ",log_dir)

    # configure dir: crlsf-pp/output,Save arguments so exact configuration can always be found
    os.makedirs(os.path.join(RESOURCE_DIR,opts.save_dir),exist_ok= True)
    with open(os.path.join(RESOURCE_DIR,opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)
    
    # Figure out what's the problem
    problem = load_problem(opts.problem)
    print("problem is:",problem)

    # Load data from load_path
    # load_path : Path to load model parameters and optimiser state
    # resume: Resume from previous checkpoint file
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('[*] Loading data from {}'.format(load_path))
        # load_data = mindspore_load_cpu(load_path)
        load_data = ms.load_checkpoint(load_path)
        import re
        opts.epoch_start = int(re.search(r"epoch-(\d+)",load_path).group(1))+1
        opts.n_epochs -= (opts.epoch_start-1)

    # Initialize model
    model_class = {
        'attention': AttentionModel,
        'attention_v2': AttentionStepModel,
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size,
        num_split=opts.num_split,
        node_size=opts.node_size
    )

    print("The model has been initialized!")
    
    # get model form model or cell 
    # model_ = get_inner_model(model)

    # Overwrite model parameters by parameters to load
    if load_data:
        ms.load_param_into_net(model,load_data)
        for name, param in load_data.items():
            print(f'Parameter name: {name}')
        print("Model parameters are loaded!")

    # using baseline method based on rollout to solve combinatorial optimization PP problem
    if opts.baseline == 'rollout':
        print("selected rollout...")
        # Baseline evaluator
        baseline = RolloutBaselinePP(model, problem, opts)
        print("rollout initialization complete ！")
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    if opts.bl_warmup_epochs > 0:
        print(opts.bl_warmup_epochs)
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)
    # 在训练过程中，优化器以当前step(epoch)为输入调用该实例，得到当前的学习率(使用decay_steps=1，达到原式子效果)
    lr_scheduler = nn.ExponentialDecayLR(learning_rate=opts.lr_model, decay_rate=opts.lr_decay,decay_steps=1,is_stair=True)

    group_params = [{'params': model.trainable_params(), 'lr': lr_scheduler}]
    if len(baseline.get_learnable_parameters()) > 0:
        group_params.append({'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic})

    optimizer = nn.Adam(group_params)
    if load_data:
        ms.load_param_into_net(optimizer,load_data)
        print("Optimizer parameters are loaded!")

    # Start the actual training loop
    val_dataset = problem.make_dataset(
        filename=os.path.join(RESOURCE_DIR,opts.val_dataset),size=opts.graph_size, num_samples=opts.val_size, distribution=opts.data_distribution,num_split=opts.num_split)

    if opts.resume:
        if "rng_state" in load_data:
            ms.set_seed(load_data["rng_state"])
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1
        
    print("采用的baseline是：", baseline)
    loss_fn = CustomReinforceLoss()
    if opts.eval_only:
        validate(model, val_dataset, opts)
    else:
        # for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
        #     train_epoch(
        #         model,
        #         optimizer,
        #         baseline,
        #         loss_fn,
        #         lr_scheduler,
        #         epoch,
        #         val_dataset,
        #         problem,
        #         tb_logger,
        #         opts
        #     )
        train_all(
            model,
            optimizer,
            baseline,
            loss_fn,
            lr_scheduler,
            val_dataset,
            problem,
            tb_logger,
            opts
        )

if __name__ == "__main__":
    run(get_options())
