import argparse

def parse_arguments():
    # 创建解析器
    parser = argparse.ArgumentParser()

    # 添加参数
    parser.add_argument('-n', '--number_of_nodes', type=int, help='Number of nodes', default=1)
    parser.add_argument('-m', '--number_of_GPU_per_node', type=int,help='number of GPU per node',default=2)
    parser.add_argument('--hidden_size',type=int, help='hidden size of the model',default=1024)
    parser.add_argument('--sequence_length', type=int, help='sequence length of the model',default=1024)
    parser.add_argument('--num_layers', type=int, help='number of layers of the model',default=24)
    parser.add_argument('--vocab_size', type=int, help='vocabulary size of the model',default=52256)
    parser.add_argument('--type', type=str, help='type of the model',default='gpt2')
    # 解析命令行参数
    args = parser.parse_args()

    return args