import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)
import numpy as np
from apss.nets.attention_model import set_decode_type
import time
import json
import pprint as pp

import multiprocessing as mp
import mindspore
import mindspore.nn.optim as optim
from mindspore import Tensor

from apss.nets.attention_model import AttentionModel
from apss.utils import load_model
import time
from collections import defaultdict

from tqdm import tqdm

import numpy as np

import mindspore.nn as nn

from .sa import amp_no_placement_strategy
from .cost_het_cluster import  get_cost_e,dp_cost,get_cost_c
from collections import defaultdict
import time
import json
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

def inference(
    M=2,
    N=2,
    hidden_size = 1024,
    sequence_length = 1024,
    num_layers = 24,
    vocab_size = 52256,
    type_model = "gpt2"
):
    with open('config.json', 'r') as f:
        config = json.load(f)
    RESOURCE_DIR = config["RESOURCE_DIR"]
    dir_path = os.path.join(RESOURCE_DIR, 'apss_main_logs')
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
# number of GPU per node, number of nodes
# parameter
    # M = 2
    # N = 2
    # # inter-node bandwidth, intra-node bandwidth
    cluster_info = {}
    for i in range(N - 1):
        cluster_info[i] = [mnp.array([10 * 1e9 / 32]).astype(mnp.float32), mnp.array([170 * 1e9 / 32]).astype(mnp.float32)]
    cluster_info[N - 1] = [mnp.array([50 * 1e9 / 32]).astype(mnp.float32), mnp.array([50 * 1e9 / 32]).astype(mnp.float32)]


    model_config = {
        "hidden_size": mnp.array([hidden_size]).astype(mnp.float32),
        "sequence_length": mnp.array([sequence_length]).astype(mnp.float32),
        "num_layers": mnp.array([num_layers]).astype(mnp.float32),
        "vocab_size": mnp.array([vocab_size]).astype(mnp.float32),
        "type": type_model
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
    # model_tmp_path = os.path.join(RESOURCE_DIR,"epoch-14.ckpt")
    checkpoint_path = "/root/APSS/checkpoint/AttentionModelV2"
    def load_all_model():
        models={}
        models[2], _ = load_model(os.path.join(checkpoint_path,"pp_30_2/pp_30_2_final.ckpt"))
        models[4],_ =  load_model(os.path.join(checkpoint_path,"pp_30_4/pp_30_4_final.ckpt"))
        models[8],_ =  load_model(os.path.join(checkpoint_path,"pp_30_8/pp_30_8_final.ckpt"))
        models[16],_=  load_model(os.path.join(checkpoint_path,"pp_30_16/pp_30_16_final.ckpt"))
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
    def pipe_rl(models, L, cost_e, cost_c, k, B):
        if k==1:
            return [cost_e.shape[0]], None
        ori_data = cost_e.view(1,-1,1)
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
        n_cost_e = cost_e/cost_e.max()#pow(10,count_e-1) * cost_e
        n_cost_c = cost_c/cost_c.max()#pow(10,count_c-1) * cost_c
        for j in range(cost_e.shape[0]-1):
            new_sample.append([sum(n_cost_e[:j+1]),sum(n_cost_e[j+1:])]+n_cost_c[j,:].tolist())
        new_data.append(new_sample)
        
        context.set_context(device_target="GPU")
        input_data = mnp.array(new_data).astype(mnp.float32)

        model = models[k]
        set_decode_type(model, "greedy")
        cost, log_likelihood, pi = model(input_data, ori_data, cost_c_data, return_pi=True)
        print(pi)
        part = pi2partition(pi[0].tolist(),cost_e.shape[0])
        time2= time.time()
        print("GNN cost: ", time2-time1, "cost: ", cost)
        return part, None

    def pipe_rl_sample(models, L, cost_e, cost_c, k, B,batchsize=1024):
        if k==1:
            return [cost_e.shape[0]], None
        ori_data = cost_e.view(1,-1,1)
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
        # n_cost_c = cost_c/cost_c.max()#pow(10,count_c-1) * cost_c
        n_cost_c = ops.ones((cost_c.shape[0],cost_c.shape[1])) * 0.5
        for j in range(cost_e.shape[0]-1):
            new_sample.append([sum(n_cost_e[:j+1]),sum(n_cost_e[j+1:])]+n_cost_c[j,:].tolist())
        new_data.append(new_sample)
        input_data = mnp.array(new_data).astype(mnp.float32)
        model = models[k]
        set_decode_type(model, "greedy")
        
        cost, log_likelihood, pi = model(input_data, ori_data, cost_c_data, return_pi=True)
        best_partition = pi2partition(pi[0].tolist(),cost_e.shape[0])
        time2= time.time()
        # print("GNN cost: ", time2-time1, "cost: ", cost)
        best_cost = pipe_cost(L, cost_e, cost_c, mindspore.tensor(k), B, best_partition)
        set_decode_type(model, "sampling")
        _, _, pis = model(input_data.tile((batchsize,1,1)), ori_data.tile((batchsize,1,1)), cost_c_data.tile((batchsize,1,1)), return_pi=True)
        for i in range(pis.shape[0]):
            part = pi2partition(pis[i,...].tolist(),cost_e.shape[0])
            cost = pipe_cost(L, cost_e, cost_c, mindspore.tensor(k), B, part)
            if cost<best_cost:
                best_partition = part
                best_cost=cost
        print("best cost:", best_cost)
        return best_partition, None

    class APSS(nn.Cell):
        def __init__(self, model_config, exp_name, placement=False):
            
            super().__init__()
            self.model_config = model_config
            #self.estimate = estimate
            self.model_type = model_config["type"]
            self.placement = placement
            # assert self.model_type == "gpt2" 
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
                known_record = f"/root/APSS/resource/known_cost/{self.model_type}_{num_layers}_{mp_size}"

        
                cur_profile_cost1 = 3 * np.load(f"{known_record}.npy")
                
                known_record = f"/root/APSS/resource/known_cost/{self.model_type}_{num_layers}_{mp_size}"

                cur_profile_cost2 = 3 * np.load(f"{known_record}.npy")

                print("cur_profile_cost1:",cur_profile_cost1)
                # average between different speed of GPUs
                cur_profile_cost = cur_profile_cost1 * 0.75 + cur_profile_cost2 * 0.25
                
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
            else:
                # partition, _ = pipe_ast(len(cost_e), np.asarray(cost_e), np.asarray(cost_c), int(pp.item()), int(B.item()))
                # partition, _ = pipe_rl(self.models, len(cost_e), cost_e, cost_c, int(pp.item()), int(B.item()))
                partition, _ = pipe_rl_sample(self.models, len(cost_e), cost_e, cost_c, int(pp.item()), int(B.item()))
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
    model = APSS(model_config, exp_name)
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

