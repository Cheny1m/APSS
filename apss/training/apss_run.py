# 放入华为云服务器运行时需取消注释下面代码
# import sys
# import subprocess

# def install_tensorboard():
#   try:
#     subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorboard_logger"])
#     print("tensorboard_logger 安装成功")
#   except subprocess.CalledProcessError as e:
#     print(f"安装 TensorBoard时发生错误：{e}")
    
# install_tensorboard()

import os
import argparse
import sys
import time
import json

import mindspore as ms

from .run_mc import run
from .options import get_options
from .generate_pp_data import generate,generate_options

with open('config.json', 'r') as f:
    config = json.load(f)
CONTEXT_MODE = config["CONTEXT_MODE"]
DEVICE_TARGET = config["DEVICE_TARGET"]
DEVICE_ID = config["DEVICE_ID"]

def main():
    parser = argparse.ArgumentParser(description="Run training with specified graph size and number of splits.")
    parser.add_argument('--graph_size', type=int, required=True, choices = [8,18,25,30,42,54,102],help='Size of the graph.')
    parser.add_argument('--num_split', type=int, required=True,choices=[1,3,7,15,31,63], help='Number of splits.')
    parser.add_argument('--model',required=True, default='attention', help="Model, 'attention' or 'attention_v2'")
    parser.add_argument('--load_path', help="ckpt path,eg:'/home/cym/APSS-main/APSS-main/resource/outputs/pp_52/pp_52_8_20240701T180050/epoch-50.ckpt'")
    parser.add_argument("--rebuild_data",action="store_true",help="If set, program will rebuild training data.")

    args = parser.parse_args()

    # 生成训练数据
    gen_opts = generate_options(['--graph_size',str(args.graph_size)])
    if args.rebuild_data:
        generate(gen_opts)
        print("New training data ready!")
    else:
        print("Using previous data!")
    sys.argv = [args for args in sys.argv if not args.startswith('--rebuild_data')]

    max_num_split = args.num_split

    # 设置训练环境，执行训练
    ms.set_context(device_target=DEVICE_TARGET, device_id = DEVICE_ID, mode=CONTEXT_MODE)

    # 批量求解
    # for num_split in [1,3,7,15,31,63]:
    #     if num_split > max_num_split:
    #         continue
    #         # 获取训练参数
    #     opts = get_options()
    #     opts.num_split = num_split
    #     opts.run_name = "{}_{}_{}_{}".format(opts.problem, opts.graph_size,opts.num_split + 1,time.strftime("%Y%m%dT%H%M%S")) 
    #     opts.node_size = opts.graph_size
    #     datadir = os.path.join(gen_opts.data_dir, opts.problem)    
    #     opts.val_dataset = os.path.join(datadir, "{}_{}_{}_{}_seed{}.pkl".format(opts.problem,opts.graph_size, opts.num_split + 1,gen_opts.name,opts.seed))
    #     opts.save_dir = os.path.join(opts.output_dir,"{}_{}".format (opts.problem, opts.graph_size),opts.run_name)
    #     run(opts)
    #     print("The training of graphsize:{},num_split:{} is finish!".format(opts.graph_size , num_split))

    # 执行某个特殊的求解
    opts = get_options()
    opts.num_split = max_num_split
    if args.load_path:
        import re
        opts.run_name = re.search(r"pp_\d+\d+\d+T\d+",args.load_path).group()
    else:
        opts.run_name = "{}_{}_{}_{}".format(opts.problem, opts.graph_size,opts.num_split + 1,time.strftime("%Y%m%dT%H%M%S")) 
    opts.node_size = opts.graph_size
    datadir = os.path.join(gen_opts.data_dir, opts.problem)    
    opts.val_dataset = os.path.join(datadir, "{}_{}_{}_{}_seed{}.pkl".format(opts.problem,opts.graph_size, opts.num_split + 1,gen_opts.name,opts.seed))
    opts.save_dir = os.path.join(opts.output_dir,"{}_{}".format (opts.problem, opts.graph_size),opts.run_name)
    run(opts)
    print("The training of graphsize:{},num_split:{} is finish!".format(opts.graph_size , max_num_split))

if __name__ == "__main__":
    main()

# 0
# python -m apss.training.apss_run --graph_size 52 --num_split 7 --model attention_v2 --load_path /home/cym/APSS-main/APSS-main/resource/outputs/pp_52/pp_52_8_20240704T151223/epoch-50.ckpt
# 1
# python -m apss.training.apss_run --graph_size 52 --num_split 1 --model attention_v2 --rebuild_data
