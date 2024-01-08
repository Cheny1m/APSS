import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# pip install tqdm
# from test import get_partiton_cost_sequence, pipe_ast,pi2partition
import numpy as np
from apss.nets.attention_model import set_decode_type
import time
import json
import pprint as pp

# import torch.multiprocessing as mp
import multiprocessing as mp
import mindspore
import mindspore.nn.optim as optim
from mindspore import Tensor
# pip install tensorboard_logger

from apss.nets.attention_model import AttentionModel
from apss.utils import load_problem, load_model, load_model_temp#,torch_load_cpu
import math
import time
from collections import defaultdict
import operator
import shutil
import random
import copy

from tqdm import tqdm

import numpy as np

import mindspore.nn as nn
# sys.path.append("/home/oj/distributed_floder/research/AMP/src/")
# sys.path.append("/root/cym/AMP/src/")

# pip install numpy
from .sa import amp_no_placement_strategy
# pip install spur
from .cost_het_cluster import  get_cost_e,dp_cost,get_cost_c
from collections import defaultdict
import time
import json
import copy

import subprocess
import sys
import os

import mindspore
import mindspore.nn as nn
import numpy as np

from .pipe import pipe_ds, pipe_ast, pipe_cost, pipe_uniform, pipe_gpt2
import mindspore.numpy as mnp
import mindspore.ops as ops
import mindspore.context as context
import numpy as np


# home_path = "/home/oj/distributed_floder/research/AMP" #os.environ['HOME']
# home_path = "/root/cym/AMP"
# home_path = '../../../data'
# dir_path = os.path.join(home_path, 'apss_main_logs')
# if not os.path.exists(dir_path):
#     os.mkdir(dir_path)
def inference():
    with open('config.json', 'r') as f:
        config = json.load(f)
    RESOURCE_DIR = config["RESOURCE_DIR"]
    dir_path = os.path.join(RESOURCE_DIR, 'apss_main_logs')
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
# number of GPU per node, number of nodes
    M = 2
    N = 2

    # # inter-node bandwidth, intra-node bandwidth
    # for i in range(N-1):
    #         cluster_info[i] = [mindspore.nmp([10 * 1e9 / 32]).float(), torch.tensor([170 * 1e9 / 32]).float()]
    # cluster_info[N-1] = [torch.tensor([50 * 1e9 / 32]).float(), torch.tensor([50 * 1e9 / 32]).float()]

    # model_config = {"hidden_size": torch.tensor([1024]).float(), 
    #                 "sequence_length": torch.tensor([1024]).float(), 
    #                 "num_layers": torch.tensor([24]).float(), 
    #                 "vocab_size":torch.tensor([52256]).float(),
    #                 "type":"gpt2"}


    cluster_info = {}

    for i in range(N - 1):
        cluster_info[i] = [mnp.array([10 * 1e9 / 32]).astype(mnp.float32), mnp.array([170 * 1e9 / 32]).astype(mnp.float32)]
    cluster_info[N - 1] = [mnp.array([50 * 1e9 / 32]).astype(mnp.float32), mnp.array([50 * 1e9 / 32]).astype(mnp.float32)]

    model_config = {
        "hidden_size": mnp.array([1024]).astype(mnp.float32),
        "sequence_length": mnp.array([1024]).astype(mnp.float32),
        "num_layers": mnp.array([24]).astype(mnp.float32),
        "vocab_size": mnp.array([52256]).astype(mnp.float32),
        "type": "gpt2"
    }


    config_h = int((model_config["hidden_size"]).item())
    config_n = int(model_config["num_layers"].item())
    time_stamp = int(time.time())
    exp_name = f"het_cluster"
    record_file = f"{os.path.join(dir_path, exp_name)}_{time_stamp}.txt"
    simulate_dir = os.path.join(RESOURCE_DIR, "apss_simulate")
    if not os.path.exists(simulate_dir):
        os.mkdir(simulate_dir)
    print("record file : ", record_file)
    print("simulate dir : ", simulate_dir)

    # remove cache directory from last run
    if os.path.exists(os.path.join(RESOURCE_DIR, "tmp")):
        for root, dirs, files in os.walk(os.path.join(RESOURCE_DIR, "tmp")):
            for f in files:
                os.unlink(os.path.join(root, f))

    # save this name to env
    os.environ["apss_log_path"] = record_file
    model_tmp_path = os.path.join(RESOURCE_DIR,"epoch-14.ckpt")
    def load_all_model():
        models={}
        # models[2], _ = load_model("./outputs/pp_30/pp30_2_rollout_20230402T234551/epoch-163.pt")
        # models[4],_ =  load_model("./outputs/pp_30/pp30_4_rollout_20230327T000146/epoch-99.pt")
        # models[8],_ = load_model("./outputs/pp_30/pp30_8_rollout_20230402T234340/epoch-134.pt")
        # models[16],_=  load_model("./outputs/pp_30/pp30_16_rollout_20230402T234155/epoch-62.pt")
        # models[2] = models[2].eval()
        # models[2] = models[2].cuda()
        # models[4] = models[4].eval()
        # models[4] = models[4].cuda()
        # models[8] = models[8].eval()
        # models[8] = models[8].cuda()
        # models[16] = models[16].eval()
        # models[16] = models[16].cuda()
        models[2], _ = load_model_temp(model_tmp_path,1)
        models[4],_ =  load_model_temp(model_tmp_path,3)
        models[8],_ = load_model_temp(model_tmp_path,7)
        models[16],_=  load_model_temp(model_tmp_path,15)
        models[2] = models[2].set_train(False)
        models[4] = models[4].set_train(False)
        models[8] = models[8].set_train(False)
        models[16] = models[16].set_train(False)
        return models
    

    def pi2partition(pi,node_size):
        pi.sort()
        # print(pi)
        assert node_size > pi[-1]+1, print(node_size,pi)
        piadd1 = [i+1 for i in pi]
        piadd1 = [0] + piadd1 + [node_size]
        partition = []
        for i, p in enumerate(piadd1):
            if i ==0:
                continue
            partition.append(p - piadd1[i-1])
        return partition
    def get_partiton_cost_sequence(data,cost_c_data,partition):
        pp=len(partition)
        s = partition
        p = [s[0]-1]


        for i in range(1, pp):
            p.append(p[i - 1] + s[i])
        lens = ops.reshape(ops.sum(data[:p[0] + 1]), (-1, 1))

        for i in range(len(s) - 1):
            lens = ops.Concat((lens, ops.reshape(ops.sum(data[p[i] + 1:p[i + 1] + 1]), (-1, 1))), axis=0)

        max_sub_seq_cost = lens.view(-1,).max()
        for i in range(pp-1):
            max_sub_seq_cost += cost_c_data[p[i]][i]
        return max_sub_seq_cost
    # partition, _ = pipe_ast(len(cost_e), np.asarray(cost_e), np.asarray(cost_c), int(pp.item()), int(B.item()))
    def pipe_rl(models, L, cost_e, cost_c, k, B):
        if k==1:
            # return [cost_e.size(0)], None
            return [cost_e.shape[0]], None
        # print(cost_e.size(),cost_e)
        # print(cost_c.size(),cost_c)
        # ori_data = cost_e.view(1,-1,1).cuda()
        ori_data = cost_e.view(1,-1,1)
        # cost_c_data = cost_c[None,...].cuda()
        cost_c_data = cost_c[None,...]
        max_c = cost_c.max()
        count_c = 0
        while max_c <1:
            count_c+=1
            max_c = max_c * 10
        max_e = cost_e.max()
        count_e = 0
        while max_e <1:
            count_e+=1
            max_e = max_e * 10
        print("count_e: ",count_e )
        print("count_c: ",count_c )
        
        time1=time.time()
        new_data = []
        new_sample = []
        # n_cost_e = pow(10,count_e-1) * cost_e
        # n_cost_c = pow(10,count_c-1) * cost_c
        n_cost_e = cost_e/cost_e.max()#pow(10,count_e-1) * cost_e
        n_cost_c = cost_c/cost_c.max()#pow(10,count_c-1) * cost_c
        # print(n_cost_e)
        # print(n_cost_c)
        # for j in range(cost_e.size(0)-1):
        for j in range(cost_e.shape[0]-1):
            new_sample.append([sum(n_cost_e[:j+1]),sum(n_cost_e[j+1:])]+n_cost_c[j,:].tolist())
        new_data.append(new_sample)
        
        # input_data =  torch.FloatTensor(new_data).cuda()
        context.set_context(device_target="GPU")
        input_data = mnp.array(new_data).astype(mnp.float32)

        model = models[k]
        set_decode_type(model, "greedy")
        cost, log_likelihood, pi = model(input_data, ori_data, cost_c_data, return_pi=True)
        # print(pi)
        # part = pi2partition(pi[0].tolist(),cost_e.size(0))
        part = pi2partition(pi[0].tolist(),cost_e.shape[0])
        time2= time.time()
        print("GNN cost: ", time2-time1, "cost: ", cost)
        return part, None
        # gnn_cots = get_partiton_cost_sequence(ori_data.view(-1),cost_c_data[0,...],part)
    # home_dir = "/home/oj/distributed_floder/research/AMP" #os.environ['HOME']
    home_dir = "/root/cym/AMP" #os.environ['HOME']

    workdir_path = os.path.join(home_dir, "AMP/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism")
    example_path = os.path.join(workdir_path, "examples")
    sys.path.append(workdir_path)
    sys.path.append(example_path)

    class AMP(nn.Cell):
        def __init__(self, model_config, exp_name, placement=False):
            
            super().__init__()
            self.model_config = model_config
            #self.estimate = estimate
            self.model_type = model_config["type"]
            self.placement = placement
            assert self.model_type == "gpt2" 
            self.init_param()
            
        def init_param(self):
            h = float(self.model_config["hidden_size"].item())
            n = float(self.model_config["num_layers"].item())
            s = float(self.model_config["sequence_length"].item())
            v = float(self.model_config["vocab_size"].item())
    
            config_h = int((self.model_config["hidden_size"]).item())
            config_n = int(n)

            self.profile_cost = {}
            #if self.estimate:
            for mp_size in [1,2,4]:
                # known_cost directory stores the real forward time with correponding model parallel degree.
                
                # known_record = f"/home/oj/distributed_floder/research/AMP/src/known_cost/{self.model_type}_P3_{mp_size}"
                # known_record = f"/root/cym/AMP/src/known_cost/{self.model_type}_P3_{mp_size}"
                known_record = f"/root/APSS/resource/known_cost/{self.model_type}_P3_{mp_size}"
                
                cur_profile_cost1 = 3 * np.load(f"{known_record}.npy")
                
                # known_record = f"/home/oj/distributed_floder/research/AMP/src/known_cost/{self.model_type}_G4_{mp_size}"
                # known_record = f"/root/cym/AMP/src/known_cost/{self.model_type}_P3_{mp_size}"
                known_record = f"/root/APSS/resource/known_cost/{self.model_type}_P3_{mp_size}"

                cur_profile_cost2 = 3 * np.load(f"{known_record}.npy")

                # average between different speed of GPUs
                cur_profile_cost = cur_profile_cost1 * 0.75 + cur_profile_cost2 * 0.25
                cur_profile_cost = cur_profile_cost[2:26]
                print("cur_profile_cost:",cur_profile_cost)

                self.profile_cost[str(mp_size)] = cur_profile_cost
                #print(f"using profile cost with mp_size {mp_size}: {cur_profile_cost}")
        
            self.models = load_all_model()
                
        def predict(self, config, bs, mbs, cluster_info, model_config, amp_config, oth):
            L = model_config["num_layers"]
            
            cost = mnp.zeros((1,))
            
            M, N = config.shape
            config = np.asarray(config)

            if np.all(config == -1):
                rank_map = defaultdict(list)
                rank_node_map = dict()

                m = oth["mp_deg"]
                n = oth["dp_deg"]
                pp = oth["pp_deg"]                   

                # infer a GPU rank map                
                counter = 0    
                for j in range(N):
                    for k in range(M):
                        # TODO: bad code here, config counts from 1
                        rank_map[j].append(counter)
                        rank_node_map[counter] = j
                        counter += 1


            # valid config, inferred from sa 
            else:
                config = config.from_numpy(config)
                pp = ops.max(config).float()

                # infer rank_map: given node name, returns the global mapped rank(int) in (pp, dp, mp) order
                # rank_node_map: given rank, returns the node
                rank_map = defaultdict(list)
                rank_node_map = dict()

                if pp >= (L + 2):
                    print(f"early return with pp={pp}, L={L}")
                    # return None, None, torch.tensor([float("inf")])
                    return None, None, mindspore.tensor([float("inf")])
                m = oth["mp_deg"]
                n = oth["dp_deg"]
                assert pp == oth["pp_deg"]                   

                rank_counter = np.zeros(int(pp.item()))

                # infer a GPU rank map                    
                for j in range(N):
                    for k in range(M):
                        # TODO: bad code here, config counts from 1
                        cur_pp = int(config[k][j] - 1)
                        rank_map[j].append(int((rank_counter[cur_pp] + cur_pp * m * n).item()))
                        rank_node_map[int((rank_counter[cur_pp] + cur_pp * m * n).item())] = j
                        rank_counter[cur_pp] += 1

            # infer number of micro-batch size B
            B = bs / (n * mbs)

            parallel_config = {"mp" : m, "dp" : n, "pp" : pp, "micro_bs" : mbs, "rank_map" : rank_map, "rank_node_map": rank_node_map}

            cost_e = get_cost_e(cluster_info=cluster_info, 
                                model_config=model_config, parallel_config=parallel_config, amp_config=amp_config)
            cost_c = get_cost_c(cluster_info=cluster_info, 
                                model_config=model_config, parallel_config=parallel_config, amp_config=amp_config)

            if int(B.item()) == 1:
                partition, _ = pipe_uniform(int(L.item()), int(pp.item()))
                # partition[0] += 2
                # partition[-1] += 4
            else:
                # partition, _ = pipe_ast(len(cost_e), np.asarray(cost_e), np.asarray(cost_c), int(pp.item()), int(B.item()))
                # partition, _ = pipe_rl_sample(self.models, len(cost_e), cost_e, cost_c, int(pp.item()), int(B.item()))
                partition, _ = pipe_rl(self.models, len(cost_e), cost_e, cost_c, int(pp.item()), int(B.item()))
                
            print(f"apss gives partition: {partition}")
            cost = pipe_cost(L, cost_e, cost_c, pp, B, partition)

            # translate to ds form, add data parallelism cost
            ds_partition, dp_side_cost = dp_cost(config, cluster_info=cluster_info, 
                                model_config=model_config, parallel_config=parallel_config, 
                                amp_config=amp_config, partition=partition)

            cost += dp_side_cost
            #print(ds_partition, cost, dp_side_cost)
            return rank_map, ds_partition, cost

    
        def construct(self, args):
            model_type = self.model_type
            config, bs, micro_bs, cluster_info, model_config, oth = args
            amp_config = {"profile_cost" : self.profile_cost}
            rank_map, partition, amp_pred = self.predict(config, bs, micro_bs, cluster_info, model_config, amp_config, oth)
            return rank_map, partition, amp_pred
    global_bs = 32
    model = AMP(model_config, exp_name)
    assert (global_bs % M == 0) and (global_bs % N == 0), "global batch size is too irrgular"

    want_simulate = [] 
    feasible = {}

    with open(record_file, "a") as fp:
        fp.write(f"{model_config}\n")                
        fp.write(f"gbs:{global_bs}\n")                
    known = None
    iter_count = 0
    time_s = time.time()
    # Estimating best configurations
    while True:
        ret = amp_no_placement_strategy(M=M, N=N, gbs=global_bs, known=known)
        if ret is None:
            break
        else:
            mp, dp, mbs, known = ret
            oth = {"mp_deg": ops.ones(1,mindspore.float32)*mp, "dp_deg": ops.ones(1,mindspore.float32)*dp, "pp_deg": ops.ones(1,mindspore.float32)*(M*N/(mp*dp))}
            fake_config = np.ones((M,N)) * (-1)
            model_args = (fake_config, global_bs, mbs, cluster_info, model_config, oth)    
            if (M*N)/(mp*dp)>30:
                continue
            # with torch.no_grad():
            rank_map, partition, cost = model(model_args)
            
            want_simulate.append(((mbs, oth, rank_map, partition), cost))
        iter_count += 1
        if iter_count % 10 == 0:
            print(f"APSS finish {iter_count} iterations")
    time_e = time.time()
    print(f"APSS finishes without placement in {iter_count} iterations in {time_e - time_s}")

    sorted_settings = sorted(want_simulate, key = lambda kv: kv[1])
    print(record_file)
    with open(record_file, "a") as fp:
        for item in sorted_settings:
            fp.write(f"rank {sorted_settings.index(item)}: {item}")
            fp.write("\n")
    return partition
partition = inference()

os.environ['PARTITION'] = partition
bash gpt2.sh