from .inference_isomorphism import inference
import os
import sys
from .get_parameter import parse_arguments
import subprocess

if __name__ == "__main__":
    args_ = parse_arguments()
print(args_)
partition = inference(M = args_.number_of_GPU_per_node,N = args_.number_of_nodes,hidden_size=args_.hidden_size,sequence_length=args_.sequence_length,num_layers=args_.num_layers,vocab_size=args_.vocab_size,type_model=args_.type)

partition_str = ','.join(map(str, partition))
os.environ['PARTITION'] = partition_str
# script_path = "" #运行mindformers的脚本路径
# subprocess.call(['sh', script_path,'2','hostfile'])