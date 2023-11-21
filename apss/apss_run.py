import os
import argparse

from run_mc import run
from options import get_options

def main():
    parser = argparse.ArgumentParser(description="Run training with specified graph size and number of splits.")
    parser.add_argument('--graph_size', type=int, required=True, help='Size of the graph')
    parser.add_argument('--num_split', type=int, required=True, help='Number of splits')

    args = parser.parse_args()

    # 获取 run_mc.py 中的默认选项
    opts = get_options()

    # 更新选项以反映传入的参数
    opts.graph_size = args.graph_size
    opts.num_split = args.num_split
    opts.node_size = opts.graph_size

    opts.val_dataset = os.path.join(os.path.join('data', 'pp'), "pp_{}_{}_validation_seed1234.pkl".format(opts.graph_size, opts.num_split))

    # 运行训练
    run(opts)

if __name__ == "__main__":
    main()
