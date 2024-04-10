import argparse
import mindspore as ms
import torch
import nets 
from apss.nets import attention_model
from apss.utils import load_problem

parser = argparse.ArgumentParser(description='Convert PyTorch model to MindSpore.')
parser.add_argument('pth_path', type=str, help='Path to the PyTorch .pth file')
parser.add_argument('ckpt_path', type=str, help='Path to save the MindSpore .ckpt file')
parser.add_argument('num_split', type=int, help='Number of splits for the MindSpore model')
parser.add_argument('node_size', type=int, help='Node size for the MindSpore model')

args = parser.parse_args()

# pth_path = "/root/APSS/torchoutputs/pp_30/pp30_2_rollout_20230402T234551/epoch-163.pt"

def pytorch_params(pth_file):
    par_dict = torch.load(pth_file, map_location=torch.device('cpu'))
    pt_params = {}
    weight_dict = par_dict["model"]
    for key,value in weight_dict.items():
        pt_params[key] = value.numpy()
    return pt_params

pt_param = pytorch_params(args.pth_path)
print("PyTorch pt is already!")

def mindspore_params(network):
    ms_params = {}
    for param in network.get_parameters():
        name = param.name
        value = param.data.asnumpy()
        ms_params[name] = value
    return ms_params

net = attention_model.AttentionModel(128, 128, load_problem('pp'), n_encode_layers=3, num_split=args.num_split, node_size=args.node_size)
ms_param = mindspore_params(net)
print("MindSpore model is already!")

def param_convert(ms_params, pt_params, ckpt_path):
    print("Start convert...")
    # 参数名映射字典
    bn_ms2pt = {"gamma": "weight",
                "beta": "bias",
                "moving_mean": "running_mean",
                "moving_variance": "running_var"}
    new_params_list = []
    for ms_param in ms_params.keys():
        # 在参数列表中，只有包含normalizer的参数是BatchNorm算子的参数
        if "normalizer" in ms_param:
            ms_param_item = ms_param.split(".")
            pt_param_item = ms_param_item[:-1] + [bn_ms2pt[ms_param_item[-1]]]
            pt_param = ".".join(pt_param_item)
            # 如找到参数对应且shape一致，加入到参数列表
            if pt_param in pt_params and pt_params[pt_param].shape == ms_params[ms_param].shape:
                ms_value = pt_params[pt_param]
                new_params_list.append({"name": ms_param, "data": ms.Tensor(ms_value)})
            else:
                print(ms_param, "not match in pt_params")
        # 其他参数
        else:
            # 如找到参数对应且shape一致，加入到参数列表
            if ms_param in pt_params and pt_params[ms_param].shape == ms_params[ms_param].shape:
                ms_value = pt_params[ms_param]
                new_params_list.append({"name": ms_param, "data": ms.Tensor(ms_value)})
            else:
                print(ms_param, "not match in pt_params")
    # 保存成MindSpore的checkpoint
    ms.save_checkpoint(new_params_list, ckpt_path)

# ckpt_path = "/root/APSS/resource/outputs/pp_30/pp_30_2_final"
param_convert(ms_param, pt_param, args.ckpt_path)
print("MindSpore ckpt had store in", args.ckpt_path)