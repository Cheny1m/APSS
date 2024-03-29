import copy
import time

# import torch
import mindspore
import mindspore.numpy as mnp
import numpy as np
import mindspore.ops.operations as ops

def pipe_ast(L, cost_e, cost_c, k, B):
    time_dp_s = time.time()
    for i in range(len(cost_e)):
        cost_e[i] = float(cost_e[i].item())
    possible = [0]
    # 用双层循环，生成所有可能的stage组合
    for i in range(1, L+1):
        ptr = 0
        while ptr + i <= L:
            possible.append(sum(cost_e[ptr:ptr+i]))
            ptr += 1
    # 初始化所有可能的m，并排序，从小到大
    possible = sorted(list(set(possible)))
    # 初始化dp[i][j][m]
    trace = []
    for i in range(L):
        outer = []
        for j in range(k):
            inner = []
            for m in range(len(possible)):
                inner.append(([],np.infty))
            outer.append(inner)
        trace.append(outer)
    # 开始动态规划
    for i in range(L):
        for j in range(k):
            for m in range(len(possible)):
                if i+1 <= j: # invalid
                    pass
                else:
                    if j == 0: # base case: 0 cut #只有一个stage的情况
                        cur_sum = sum(cost_e[:i+1])
                        assert cur_sum in possible
                        trace[i][j][m] = ([i+1], (B-1) * max(0, cur_sum - possible[m]))
                    else: #
                        cost_best = np.infty
                        S_best = []
                        for cut in range(j-1, i):
                            cur_sum = sum(cost_e[cut+1:i+1])
                            assert cur_sum in possible
                            S, cost_ = trace[cut][j-1][possible.index(max(cur_sum, possible[m]))]
                            cost_ += (B-1) * max(0, cur_sum - possible[m])
                            cost_ += cost_c[cut][j-1]
                            if cost_ < cost_best:
                                cost_best = cost_ - cost_c[cut][j-1]
                                S_ = copy.deepcopy(S)
                                S_.append(i-cut)
                                S_best = S_
                        trace[i][j][m] = (S_best, cost_best)
                        
    time_dp_used = time.time() - time_dp_s
    
    # add each stage cost at the end 
    S, cost = trace[L-1][k-1][0]
    cost += np.sum(cost_e)
    print(f"pipe_ast used {round(time_dp_used,2)} seconds with {L} layers and {k} stages.")
    return (S, cost)

def pipe_ds(L, cost_e, cost_c, k, B):
    per_stage = L // k
    s = [int(per_stage.item())] * (int(k.item())-1)
    # s = [int(per_stage)] * (int(k)-1)
    s.append(int(L.item())-sum(s))
    # s.append(int(L)-sum(s))
    p = [s[0]-1]
    
    for i in range(1, int(k.item())):
    # for i in range(1, int(k)):
        p.append(p[i-1] + s[i])
    # lens = torch.reshape(torch.sum(cost_e[:p[0]+1]), (-1,1))
    lens = mnp.reshape(mnp.sum(cost_e[:p[0] + 1]), (-1, 1))
    
    for i in range(len(s)-1):
        # lens = torch.cat([lens,torch.reshape(torch.sum(cost_e[p[i]+1:p[i+1]+1]), (-1,1))])
        lens = ops.Concat((lens, mnp.reshape(mnp.sum(cost_e[p[i] + 1:p[i + 1] + 1]), (-1, 1))), axis=0)
        
    # max_l = torch.max(lens)
    max_l = mnp.max(lens)
    cost = (B-1) * max_l
    for i in range(int(k.item())-1):
    # for i in range(int(k)-1):
        cost += cost_c[p[i]][i]
    # cost += torch.sum(cost_e)
    cost += mnp.sum(cost_e)
    return s, cost

def pipe_gpt2(L, pp):
    each = L // pp
    remain = L - pp * each
    start = 2
    ret = [start + each]
    for i in range(pp-1):
        ret.append(each)
    for i in range(remain):
        ret[i] += 1
    ret[-1] += 4
    #print(f"-----------{ret}. {L}, {pp}")
    return ret, None

def pipe_uniform(L, pp):
    #print("using a uniform")
    each = L // pp
    remain = L - pp * each
    ret = [each]
    for i in range(pp-1):
        ret.append(each)
    for i in range(remain):
        ret[i] += 1
    # print(f"pipe uniform returns {ret}")
    # print(f"-----------{ret}. {L}, {pp}")
    return ret, None

def pipe_transgan(cost_e, pp):
    # buggy
    assert False
    each = np.sum(cost_e) // pp
    assignment = []
    cumulative_time = 0
    cumulative_length = 0
    print(cost_e)
    print(each)
    for i in range(len(cost_e)):
        cumulative_time += cost_e[i]
        cumulative_length += 1
        if cumulative_time >= each:
            assignment.append(cumulative_length)
            cumulative_time = 0
            cumulative_length = 0
        print(cumulative_time, cumulative_length)
    #remain = 
    if cumulative_length != 0:
        assignment.append(cumulative_length)
    return assignment, None

def pipe_cost(L, cost_e, cost_c, k, B, partition):
    s = partition
    # print(partition)
    p = [s[0]-1]
    
    # print("k:",int(k))
    for i in range(1, int(k)):
        p.append(p[i-1] + s[i])
    # lens = torch.reshape(torch.sum(cost_e[:p[0]+1]), (-1,1))
    lens = mnp.reshape(mnp.sum(cost_e[:p[0] + 1]), (-1, 1))
    #print(f"calculating cost: {cost_e} {cost_c} {k} {B} {partition}")
    for i in range(len(s)-1):
        # lens = torch.cat([lens,torch.reshape(torch.sum(cost_e[p[i]+1:p[i+1]+1]), (-1,1))])
        # print("ops.Concat的操作数：",mnp.reshape(mnp.sum(cost_e[p[i] + 1:p[i + 1] + 1]), (-1, 1)))
        # lens = ops.Concat(axis=0)(lens,ops.Reshape()(ops.ReduceSum()(cost_e[p[i] + 1:p[i + 1] + 1]),(-1,1)))
        lens = ops.Concat(axis=0)((lens, mnp.reshape(mnp.sum(cost_e[p[i] + 1:p[i + 1] + 1]), (-1, 1))))

    # max_l = torch.max(lens)
    max_l = mnp.max(lens)
    cost = (B-1) * max_l
    for i in range(int(k)-1):
        cost += cost_c[p[i]][i]
    # cost += torch.sum(cost_e)
    cost += mnp.sum(cost_e)
    return cost
