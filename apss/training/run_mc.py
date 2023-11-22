import os
import json
import pprint as pp
from tensorboard_logger import Logger as TbLogger

import mindspore as ms
import mindspore.nn as nn
import mindspore.communication as communication
from mindspore.communication import init,get_rank,get_group_size

from apss.nets.attention_model import AttentionModel
from apss.utils import load_problem

from .options import get_options
from .train_mc import train_epoch, validate
from .reinforce_baselines_pp import  RolloutBaselinePP,WarmupBaseline,NoBaseline

with open('config.json', 'r') as f:
    config = json.load(f)
RESOURCE_DIR = config["RESOURCE_DIR"]

def run(opts):

    # Pretty print the run args
    print("The run args is:")
    pp.pprint(vars(opts))

    # Set the random seed
    ms.set_seed(opts.seed)

    # Set the device，PYNATIVE_MODE
    device_target = "GPU" if opts.use_cuda else "CPU"
    ms.set_context(device_target=device_target, device_id = 1, mode=ms.PYNATIVE_MODE)
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

    # Initialize model
    model_class = {
        'attention': AttentionModel
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

    # Load baseline from data, make sure script is called with same type of baseline
    # if 'baseline' in load_data:
    #     print('baseline' in load_data)
    #     baseline.load_state_dict(load_data['baseline'])


    # Initialize learning rate scheduler, decay by lr_decay once per epoch!

    # # lr_scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    # def learning_rate_function(lr_decay,epoch):
    #     return lr_decay ** epoch
    # lr_scheduler = LearningRateScheduler(learning_rate_function(opts.lr_decay,epoch))
    
    # 在训练过程中，优化器以当前step(epoch)为输入调用该实例，得到当前的学习率(使用decay_steps=1，达到原式子效果)
    lr_scheduler = nn.ExponentialDecayLR(learning_rate=opts.lr_model, decay_rate=opts.lr_decay,decay_steps=1,is_stair=True)
    # lr_scheduler = lr_scheduler(epoch)
    # optimizer.group_lr = lr_scheduler
    # print("optimizer:",optimizer.group_lr.learning_rate)

    # Initialize optimizer
    # params = [{'params': model.trainable_params(), 'lr': opts.lr_model}]
    # if len(baseline.get_learnable_parameters()) > 0:
    #     params += [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
    # optimizer = optim.Adam(params=params)

    group_params = [{'params': model.trainable_params(), 'lr': lr_scheduler}]
    if len(baseline.get_learnable_parameters()) > 0:
        group_params.append({'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic})

    optimizer = nn.Adam(group_params)
    if load_data:
        ms.load_param_into_net(optimizer,load_data)
        print("Optimizer parameters are loaded!")

    # Load optimizer state
    # if 'optimizer' in load_data:
    #     param_dict = load_checkpoint(load_path)
    #     optimizer.load_state_dict(param_dict['optimizer'])
    #     for state in optimizer.get_states():
    #         for k, v in state.items():
    #             if isinstance(v, ms.Tensor):
    #                 state[k] = v

    # Start the actual training loop
    val_dataset = problem.make_dataset(
        filename=os.path.join(RESOURCE_DIR,opts.val_dataset),size=opts.graph_size, num_samples=opts.val_size, distribution=opts.data_distribution,num_split=opts.num_split)

    if opts.resume:
        if "rng_state" in load_data:
            ms.set_seed(load_data["rng_state"])
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])
        # set_seed(load_data['rng_state'])
        # if opts.use_gpu:
        #     ms.set_seed(load_data['cuda_rng_state'][0])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)

        # baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1
        
    print("采用的baseline是：", baseline)
    
    if opts.eval_only:
        validate(model, val_dataset, opts)
    else:
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            train_epoch(
                model,
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                val_dataset,
                problem,
                tb_logger,
                opts
            )

if __name__ == "__main__":
    run(get_options())