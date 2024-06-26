from collections import defaultdict
import time
import json
import copy

import subprocess
import sys
import os

import mindspore
import mindspore.nn as nn
import mindspore.ops
import numpy as np

from amp_utils import rank2axis, axis2rank, get_host
from pipe import pipe_ds, pipe_ast, pipe_cost, pipe_uniform, pipe_gpt2

home_dir = os.environ['HOME'] 
workdir_path = os.path.join(home_dir, "AMP/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism")
example_path = os.path.join(workdir_path, "examples")
sys.path.append(workdir_path)
sys.path.append(example_path)

class AMP(nn.Cell):
    def __init__(self, model_config, exp_name):
        
        super().__init__()
        self.model_config = model_config
        self.model_type = model_config["type"]
        assert self.model_type == "transgan" 
        self.init_param()
        
    def init_param(self):
        h = float(self.model_config["hidden_size"].item())
        n = float(self.model_config["num_layers"].item())
        s = float(self.model_config["sequence_length"].item())
        v = float(self.model_config["vocab_size"].item())
 
        json_path = os.path.join(example_path, "ds_config.json")

        self.profile_cost = {}
        for mp_size in [1,2,4]:
            # known_cost directory stores the real forward time with correponding model parallel degree.
            known_record = f"known_cost/{self.model_type}_P3_{mp_size}"
            # We model backward time as forward time * 2
            cur_profile_cost = 3 * np.load(f"{known_record}.npy")
            self.profile_cost[str(mp_size)] = cur_profile_cost
            #print(f"using exec cost with mp_size {mp_size}: {cur_profile_cost}")
        
    def forward(self, args):
        model_type = self.model_type
        config, bs, micro_bs, cluster_info, model_config, oth = args
        amp_config = {"profile_cost" : self.profile_cost}
        rank_map, partition, amp_pred = predict(config, bs, micro_bs, cluster_info, model_config, amp_config, oth)
        return rank_map, partition, amp_pred
        
# pipeline communication cost, return shape: (L-1, pp-1)
def get_cost_c(cluster_info, model_config, parallel_config, amp_config, dp_index=0):
    mindspore.set_context(pynative_synchronize=True)
    h = model_config["hidden_size"]
    s = model_config["sequence_length"]
    n = model_config["num_layers"]
    v = model_config["vocab_size"]
    bs = parallel_config["micro_bs"]
    rank_map = parallel_config["rank_map"]
    rank_node_map = parallel_config["rank_node_map"]
    mp = parallel_config["mp"]
    dp = parallel_config["dp"]
    pp = parallel_config["pp"]
    
    depth = model_config["depth"]
    bottom = model_config["bottom"]
    _num_layer = 1+depth[0] + depth[1] + depth[2] + depth[3] + depth[4] + depth[5]
    layer_volume = [bs * h * bottom ** 2] * _num_layer
    # Build communication cost between pipeline stages by looking up the cluster information
    cost_c = mindspore.ops.Zeros()((int(dp.item()), _num_layer-1, int(pp.item()-1)),mindspore.float32)
    for i in range(int(dp.item())):    
        for j in range(int(pp.item()-1)):
            # get the slowest mp gpu connection
            slowest_bandwidth = np.inf
            for k in range(int(mp.item())):    
                rank_cur = axis2rank(axis=(j,i,k), mp_deg=mp, dp_deg=dp, pp_deg=pp)
                rank_peer = axis2rank(axis=(j+1,i,k), mp_deg=mp, dp_deg=dp, pp_deg=pp)
                node_cur = rank_node_map[int(rank_cur.item())]
                node_peer = rank_node_map[int(rank_peer.item())]
                
                if node_cur != node_peer: 
                    cur_bandwidth = min(cluster_info[node_cur][0], cluster_info[node_peer][0])
                else:
                    cur_bandwidth = cluster_info[node_cur][1]
                if cur_bandwidth < slowest_bandwidth:
                    slowest_bandwidth = cur_bandwidth
            for k in range(_num_layer-1):
                # cost_c[i][k][j] = layer_volume[k]  / slowest_bandwidth
                cost_c[i][k][j] = mindspore.ops.squeeze(layer_volume[k]  / slowest_bandwidth)

    cost_c = mindspore.ops.ReduceMean()(cost_c, axis=0)
    # print("cost_c:",cost_c)


    return cost_c

# execution cost for one layer, return shape (L,)
def get_cost_e(cluster_info, model_config, parallel_config, amp_config):    

    h = model_config["hidden_size"]
    s = model_config["sequence_length"]
    n = model_config["num_layers"]
    v = model_config["vocab_size"]
    orig_bs = parallel_config["micro_bs"]
    rank_map = parallel_config["rank_map"]
    rank_node_map = parallel_config["rank_node_map"]
    mp = parallel_config["mp"]
    dp = parallel_config["dp"]
    pp = parallel_config["pp"]

    profile_cost = amp_config["profile_cost"]
    depth = model_config["depth"]
    bottom = model_config["bottom"] 
    
    # Only do DP for transformer layer, skip the lambda layer for faster pipeline assignment
    cost_e = [None] * int(dp.item())
    for i in range(int(dp.item())):
        cost_e[i] = []

    for i in range(int(dp.item())):
            # mp_avg is only used with placement ablation study. Ignore it in reproducing main results.
            # cost_e in the main result is equivalent to using profile_cost.

            mp_avg = mindspore.ops.Zeros()(1,mindspore.float32)
            for j in range(int(pp.item())):
                slowest = float("inf")
                for k in range(int(mp.item())):
                    rank_cur = axis2rank(axis=(j,i,k), mp_deg=mp, dp_deg=dp, pp_deg=pp)
                    node_cur = rank_node_map[int(rank_cur.item())]

                    rank_next = axis2rank(axis=(j,i,(k+1)%(mp.item())), mp_deg=mp, dp_deg=dp, pp_deg=pp)
                    node_next = rank_node_map[int(rank_next.item())]

                    if node_cur == node_next:
                        connectivity = cluster_info[node_cur][1]
                    else:
                        connectivity = min(cluster_info[node_cur][0], cluster_info[node_next][0])
                slowest = min(slowest, connectivity)
                #mp_avg += 2 * (mp-1) / (mp * slowest)

            mp_avg = 0

            bs = orig_bs
            cost_e[i].append(orig_bs * profile_cost[str(int(mp.item()))][0])
            # print(type(orig_bs * profile_cost[str(int(mp.item()))][0]))
            # cost_e[i].append(mindspore.Tensor(orig_bs * profile_cost[str(int(mp.item()))][0],mindspore.float32))
            s = bottom**2
            first_layer_id = 3

            for inc in range(depth[0]):
                cur_layer_id = first_layer_id + inc
                cur_layer = orig_bs * (profile_cost[str(int(mp.item()))][cur_layer_id])
                cur_layer += ((7*h**2/mp + 2*s*bs*h) * mp_avg).item()
                cost_e[i].append(np.float32(cur_layer))

                # print(type(np.float64(cur_layer.asnumpy().item())))

            s = 4*bottom**2
            first_layer_id = 7 + depth[0]
            for inc in range(depth[1]):
                cur_layer_id = first_layer_id + inc
                cur_layer = orig_bs * (profile_cost[str(int(mp.item()))][cur_layer_id])
                cur_layer +=  ((7*h**2/mp + 2*s*bs*h) * mp_avg).item()
                cost_e[i].append(cur_layer)

            s = 16*bottom**2
            first_layer_id = 9 + depth[0] + depth[1]
            for inc in range(depth[2]):
                cur_layer_id = first_layer_id + inc
                cur_layer = orig_bs * (profile_cost[str(int(mp.item()))][cur_layer_id])
                cur_layer += ((7*h**2/mp + 2*s*bs*h) * mp_avg).item()
                cost_e[i].append(cur_layer)

            bs *= 16
            s = 4*bottom**2
            first_layer_id = 15 + depth[0] + depth[1] + depth[2]
            for inc in range(depth[3]):
                cur_layer_id = first_layer_id + inc
                cur_layer =  orig_bs* (profile_cost[str(int(mp.item()))][cur_layer_id])
                cur_layer += ((7*(h/4)**2/mp + 2*s*bs*(h/4)) * mp_avg).item()
                cost_e[i].append(cur_layer)

            bs *= 4
            s = 4*bottom**2
            first_layer_id = 22 + depth[0] + depth[1] + depth[2] + depth[3]
            for inc in range(depth[4]):
                cur_layer_id = first_layer_id + inc
                cur_layer =  orig_bs * (profile_cost[str(int(mp.item()))][cur_layer_id])
                cur_layer += ((7*(h/16)**2/mp + 2*s*bs*(h/16)) * mp_avg).item()
                cost_e[i].append(cur_layer)

            bs *= 4
            s = 4*bottom**2
            first_layer_id = 29 + depth[0] + depth[1] + depth[2] + depth[3] + depth[4]
            for inc in range(depth[5]):
                cur_layer_id = first_layer_id + inc
                cur_layer = orig_bs * (profile_cost[str(int(mp.item()))][cur_layer_id])
                cur_layer += ((7*(h/64)**2/mp + 2*s*bs*(h/64)) * mp_avg).item()
                cost_e[i].append(np.float32(cur_layer))

            assert len(cost_e[i]) == 1 + depth[0] + depth[1] + depth[2] + depth[3] + depth[4] + depth[5]
            cost_e[i] = np.asarray(cost_e[i])

    
    cost_e = mindspore.Tensor.from_numpy(np.stack(cost_e, axis=0))            
    cost_e = mindspore.ops.ReduceMean()(cost_e, axis=0)
    return cost_e

def dp_cost(config, cluster_info,model_config, parallel_config, amp_config, partition):
    h = model_config["hidden_size"]
    s = model_config["sequence_length"]
    n = model_config["num_layers"]
    v = model_config["vocab_size"]
    bs = parallel_config["micro_bs"]
    rank_map = parallel_config["rank_map"]
    rank_node_map = parallel_config["rank_node_map"]
    mp = parallel_config["mp"]
    dp = parallel_config["dp"]
    pp = parallel_config["pp"]
   
    depth = model_config["depth"]
    # First translate to deepspeed partition form
    work_layers = [0]
    h_map = dict()
    first_layer_id = 3
    for inc in range(depth[0]):
        cur_layer_id = first_layer_id + inc
        work_layers.append(cur_layer_id)
        h_map[str(cur_layer_id)] = h

    first_layer_id = 7 + depth[0]
    for inc in range(depth[1]):
        cur_layer_id = first_layer_id + inc
        work_layers.append(cur_layer_id)
        h_map[str(cur_layer_id)] = h

    first_layer_id = 9 + depth[0] + depth[1]
    for inc in range(depth[2]):
        cur_layer_id = first_layer_id + inc
        work_layers.append(cur_layer_id)
        h_map[str(cur_layer_id)] = h

    first_layer_id = 15 + depth[0] + depth[1] + depth[2]
    for inc in range(depth[3]):
        cur_layer_id = first_layer_id + inc
        work_layers.append(cur_layer_id)
        h_map[str(cur_layer_id)] = h/4

    first_layer_id = 22 + depth[0] + depth[1] + depth[2] + depth[3]
    for inc in range(depth[4]):
        cur_layer_id = first_layer_id + inc
        work_layers.append(cur_layer_id)
        h_map[str(cur_layer_id)] = h/16

    first_layer_id = 29 + depth[0] + depth[1] + depth[2] + depth[3] + depth[4]
    for inc in range(depth[5]):
        cur_layer_id = first_layer_id + inc
        work_layers.append(cur_layer_id)
        h_map[str(cur_layer_id)] = h/64

    ptr = -1
    ds_partition = [0]
    #print(partition, work_layers)
    for i in partition:
        ptr += i
        last_layer_id = work_layers[ptr]
        ds_partition.append(last_layer_id + 1)
    ds_partition[-1] = 32 + depth[0] + depth[1] + depth[2] + depth[3] + depth[4] + depth[5]
    assert len(ds_partition) == pp + 1
                
    # should be per-dp_group time
    max_dp = mindspore.Tensor([0.])
    for i in range(int(pp.item())):
        for j in range(int(mp.item())):
            
            slowest = float("inf")
            for k in range(int(dp.item())):
                rank_cur = axis2rank(axis=(i,k,j), mp_deg=mp, dp_deg=dp, pp_deg=pp)
                node_cur = rank_node_map[int(rank_cur.item())]
                    
                rank_next = axis2rank(axis=(i,(k+1)%(dp.item()),j), mp_deg=mp, dp_deg=dp, pp_deg=pp)
                node_next = rank_node_map[int(rank_next.item())]
                    
                if node_cur == node_next:
                    connectivity = cluster_info[node_cur][1]
                else:
                    connectivity = min(cluster_info[node_cur][0], cluster_info[node_next][0])
                        
            slowest = min(slowest, connectivity)
            dp_const = 2 * (dp-1) / (dp * slowest)
            # dp_const = mindspore.Tensor([dp_const])
            dp_const = mindspore.Tensor([dp_const.item()])
            # param_count = mindspore.ops.Zeros()(1,)
            param_count = mindspore.Tensor([0.])
            for layer_id in range(ds_partition[i], ds_partition[i+1]):
                if str(layer_id) in h_map:
                    param_count += 12 * h_map[str(layer_id)] ** 2 / mp

            #print(f"dp: {dp_const} and param {param_count}")
            cur_dp = dp_const * param_count
            if cur_dp > max_dp:
                max_dp = cur_dp
                
    return ds_partition, max_dp

def predict(config, bs, mbs, cluster_info, model_config, amp_config, oth):
    L = model_config["num_layers"]
    cost = mindspore.ops.Zeros(1,)
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
        config = mindspore.Tensor.from_numpy(config)
        _,pp = mindspore.ops.Argmax()(config).float()
        
        # infer rank_map: given node name, returns the global mapped rank(int) in (pp, dp, mp) order
        # rank_node_map: given rank, returns the node
        rank_map = defaultdict(list)
        rank_node_map = dict()
    
        if pp >= (L + 2):
            print(f"early return with pp={pp}, L={L}")
            return None, None, mindspore.Tensor([float("inf")])
           
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
           
    #partition, _ = pipe_dp(int(L.item()), np.asarray(cost_e.detach()), np.asarray(cost_c.detach()), int(pp.item()), int(B.item()))
    if int(B.item()) == 1:
        partition, _ = pipe_uniform(int(L.item()), int(pp.item()))
    else:
        partition, _ = pipe_ast(len(cost_e), np.asarray(cost_e), np.asarray(cost_c), int(pp.item()), int(B.item()))
    
    print(f"amp gives partition: {partition}")
    cost = pipe_cost(L, cost_e, cost_c, pp, B, partition)
        
    # translate to ds form, add data parallelism cost
    ds_partition, dp_side_cost = dp_cost(config, cluster_info=cluster_info, 
                        model_config=model_config, parallel_config=parallel_config, 
                        amp_config=amp_config, partition=partition)
       
    cost += dp_side_cost
    #print(ds_partition, cost, dp_side_cost)
    return rank_map, ds_partition, cost
