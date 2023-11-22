import os
import argparse
import sys

from .run_mc import run
from .options import get_options
from .generate_pp_data import generate,generate_options

def main():
    parser = argparse.ArgumentParser(description="Run training with specified graph size and number of splits.")
    parser.add_argument('--graph_size', type=int, required=True, choices = [8,18,25,30,42,54,102],help='Size of the graph.')
    parser.add_argument('--num_split', type=int, required=True,choices=[1,3,7,15,31,63], help='Number of splits.')
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

    # 获取训练参数
    opts = get_options()
    opts.node_size = opts.graph_size

    datadir = os.path.join(gen_opts.data_dir, opts.problem)
    opts.val_dataset = os.path.join(datadir, "{}_{}_{}_{}_seed{}.pkl".format(opts.problem,opts.graph_size, opts.num_split + 1,gen_opts.name,opts.seed))

    # 执行训练
    for num_split in [1,3,7,15,31,63]:
        if num_split > opts.num_split:
            continue
        run(opts)
        print("The training of graphsize:{},num_split:{} is finish!".format(opts.graph_size , num_split))

if __name__ == "__main__":
    main()
