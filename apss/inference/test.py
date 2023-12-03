import time
import copy

import mindspore as ms
import mindspore.ops as ops
import mindspore.numpy as np
from mindspore import Tensor, context

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

def generate_pp_data(dataset_size=1, pp_size=4,n_split=3):
    data = np.random.uniform(size=(dataset_size, pp_size, 1)).tolist()
    new_data = []
    cost_c_data = []
    for i, sample in enumerate(data):
        new_sample = []
        elements = [x[0] for x in sample]
        print(elements)
        cost_c = np.random.uniform(size=(pp_size-1, n_split))
        cost_c_data.append(cost_c)
        for j in range(len(elements)-1):
            new_sample.append([sum(elements[:j+1]),sum(elements[j+1:])]+cost_c[j,:].tolist())
        new_data.append(new_sample)
    return new_data,data,cost_c_data


def get_partiton_cost_sequence(data, cost_c_data, partition):
    pp = len(partition)
    s = partition
    p = [s[0]-1]

    for i in range(1, pp):
        p.append(p[i-1] + s[i])
    lens = ops.reshape(ops.reduce_sum(data[:p[0]+1]), (-1,1))
    for i in range(len(s)-1):
        lens = ops.concat([lens, ops.reshape(ops.reduce_sum(data[p[i]+1:p[i+1]+1]), (-1,1))])
    max_sub_seq_cost = ops.reduce_max(lens.reshape(-1,))
    for i in range(pp-1):
        max_sub_seq_cost += cost_c_data[p[i]][i]
    return max_sub_seq_cost

def pipe_ast(cost_e, cost_c, B):
    L = cost_e.shape[0]
    k = B
    time_dp_s = time.time()
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
                                cost_best = cost_ #- cost_c[cut][j-1]
                                S_ = copy.deepcopy(S)
                                S_.append(i-cut)
                                S_best = S_
                        trace[i][j][m] = (S_best, cost_best)
                        
    time_dp_used = time.time() - time_dp_s
    
    # add each stage cost at the end 
    S, cost = trace[L-1][k-1][0]
    # cost += np.sum(cost_e)
    print(f"pipe_ast used {round(time_dp_used,2)} seconds with {L} layers and {k} stages.")
    return (S, cost)


context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
# def test(model, node_size, n_split):
#     model.set_train(False) # 测试模式

#     # 生成数据 
#     data, ori_data, cost_c_data = generate_pp_data(1, node_size, n_split)

#     data = ms.Tensor(data, ms.float32)
#     ori_data = ms.Tensor(ori_data, ms.float32)
#     cost_c_data = ms.Tensor(cost_c_data, ms.float32)

#     print("测试数据:", data.shape, ori_data.shape, cost_c_data.shape)
#     cost, log_likelihood, pi = model(data, ori_data, cost_c_data, return_pi=True)

#     part = pi2partition(pi[0].tolist(),node_size)
#     gnn_cots = get_partiton_cost_sequence(ori_data.view(-1), cost_c_data[0,...], part)
#     print("GNN结果:", part, cost.asnumpy(), gnn_cots.max().asnumpy())
#     print(n_split + 1)

#     dp, _ = pipe_ast(cost_e=ori_data.view(-1), cost_c=cost_c_data[0,...], B=n_split + 1)
#     dp_cots = get_partiton_cost_sequence(ori_data.view(-1), cost_c_data[0,...], dp)
#     print("动态规划结果:", dp, dp_cots.max().asnumpy())  

def test(model, node_size, n_split):
    model.eval()
    # while True:
    data, ori_data, cost_c_data = generate_pp_data(1, node_size, n_split)
    data = Tensor(data, ms.float32).to("cuda")
    ori_data = Tensor(ori_data, ms.float32).to("cuda")
    cost_c_data = Tensor(cost_c_data, ms.float32).to("cuda")
    print("测试数据：", data.shape, ori_data.shape, cost_c_data.shape)
    cost, log_likelihood, pi = model(data, ori_data, cost_c_data, return_pi=True)
    part = pi2partition(pi.asnumpy()[0].tolist(), node_size)
    gnn_cots = get_partiton_cost_sequence(ori_data.view(-1), cost_c_data.asnumpy()[0,...], part)
    print("GNN结果：", part, cost.asnumpy(), gnn_cots.asnumpy().max())
    print(n_split+1)
    dp, _ = pipe_ast(cost_e=ori_data.view(-1).asnumpy(), cost_c=cost_c_data.asnumpy()[0,...], B=n_split+1)
    dp_cots = get_partiton_cost_sequence(ori_data.view(-1), cost_c_data.asnumpy()[0,...], dp)
    print("动态规划结果：", dp, dp_cots.asnumpy().max())