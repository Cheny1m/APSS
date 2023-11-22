import os
import pickle
import numpy as np

from mindspore import Tensor
import mindspore as ms
import mindspore.ops as ops

from apss.utils.beam_search import beam_search

from.state_pp import StatePP,initialize_pp_state

# @ms.jit
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

# @ms.jit
def get_partition_cost_sequence(data, cost_c_data, partition):
    # print("data:",data,type(data),data.dtype,"cost_c:",cost_c_data,"partition:",partition)
    pp = len(partition)
    s = partition
    p = [s[0] - 1]
    for i in range(1, pp):
        p.append(p[i - 1] + s[i])
    lens = ops.reshape((data[:p[0] + 1]).sum(),(-1,1))
    for i in range(len(s) - 1):
        lens = ops.concat([lens,ops.reshape((data[p[i] + 1:p[i + 1] + 1]).sum(),(-1,1))])
    max_sub_seq_cost = lens.reshape(-1,).max()
    for i in range(pp - 1):
        max_sub_seq_cost += cost_c_data[p[i]][i]
    return max_sub_seq_cost

# @ms.jit
def get_pp_costs(ori_dataset, cost_c_dataset, dataset, pi):
    node_size = dataset.shape[1] + 1  # 这里输入的node数量是nodesize-1数量
    costs = []
    for idx in range(pi.shape[0]):
        position = pi[idx, :]
        position = position.asnumpy().tolist()
        data = ori_dataset[idx, :, 0]
        cost_data = cost_c_dataset[idx, ...]
        partition = pi2partition(position, node_size) 
        costs.append(get_partition_cost_sequence(data, cost_data, partition))

    costs_np= np.array([item.asnumpy() for item in costs])
    costs_tensor = Tensor(costs_np)[:, None]
    # costs = Tensor(costs)[:,None]
    
    # return costs, None
    return costs_tensor, None

def make_pp_dataset(*args, **kwargs):
    return PPDataset(*args, **kwargs)

def make_pp_state(*args, **kwargs):
    return initialize_pp_state(*args, **kwargs)

# 使用beam搜索进行策略搜索
def beam_pp_search(input, beam_size, expand_size=None, compress_mask=False, model=None, max_calc_batch_size=4096):
    assert model is not None, "Provide model"
    fixed = model.precompute_fixed(input)
    def propose_expansions(beam):
        return model.propose_expansions(
            beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
        )
    state = PP.make_state(
        input, visited_dtype=ms.int64 if compress_mask else ms.uint8
    )
    return beam_search(state, beam_size, propose_expansions)

# DPSN的数据生成
class PP(object):
    NAME = 'pp'

    # 计算并返回成本
    @staticmethod
    def get_costs(ori_dataset, cost_c_dataset, dataset, pi):
        node_size = dataset.shape[1] + 1  # 这里输入的node数量是nodesize-1数量
        costs = []
        for idx in range(pi.shape[0]):
            position = pi[idx, :]
            position = position.asnumpy().tolist()
            data = ori_dataset[idx, :, 0]
            cost_data = cost_c_dataset[idx, ...]
            partition = pi2partition(position, node_size) 
            costs.append(get_partition_cost_sequence(data, cost_data, partition))
            
        costs_np= np.array([item.asnumpy() for item in costs])
        costs_tensor = Tensor(costs_np)[:, None]
        # costs_tensor = Tensor(costs)[:,None]
        return costs_tensor, None

    # 返回一个PPDataset实例
    @staticmethod
    def make_dataset(*args, **kwargs):
        return PPDataset(*args, **kwargs)

    # make_state: 初始化并返回一个StatePP实例。
    @staticmethod
    def make_state(*args, **kwargs):
        return StatePP.initialize(*args, **kwargs)

    # 使用beam搜索进行策略搜索
    @staticmethod
    def beam_search(input, beam_size, expand_size=None, compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"
        fixed = model.precompute_fixed(input)
        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )
        state = PP.make_state(
            input, visited_dtype=ms.int64 if compress_mask else ms.uint8
        )
        return beam_search(state, beam_size, propose_expansions)

# 转换原始数据
# mindspore通过每次调用Python层自定义的Dataset以生成数据集，而Pytorch一样自定义数据集的抽象类，然后进行继承
# class PPDataset:
#     def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None, num_split=3):
#         super(PPDataset, self).__init__()

#         self.data_set = []
#         if filename is not None:
#             assert os.path.splitext(filename)[1] == '.pkl'

#             with open(filename, 'rb') as f:
#                 data = pickle.load(f)
#                 self.data = [ms.Tensor(row, dtype=ms.float32) for row in (data[offset:offset+num_samples])]
#             filename2 = filename.split('/')
#             filename2[-1] = 'ori_' + filename2[-1]
#             filename2 = '/'.join(filename2)
#             filename3 = filename.split('/')
#             filename3[-1] = 'cost_c_' + filename3[-1]
#             filename3 = '/'.join(filename3)
#             with open(filename2, 'rb') as f:
#                 data = pickle.load(f)   
#                 self.ori_data = [ms.Tensor(row, dtype=ms.float32) for row in (data[offset:offset+num_samples])]
#             with open(filename3, 'rb') as f:
#                 data = pickle.load(f)   
#                 self.cost_c_data = [ms.Tensor(row, dtype=ms.float32) for row in (data[offset:offset+num_samples])]
#         else:
#             matrix_left = mnp.zeros((size-1, size))
#             matrix_right = mnp.zeros((size-1, size))
#             for i in range(size-1):
#                 matrix_left[i, :i+1] = 1.
#                 matrix_right[i, i+1:] = 1.
#             ori_data = [Tensor(np.random.uniform(0, 1, (size, 1)),dtype = ms.float32) for _ in range(num_samples)]
#             cost_c_data = [Tensor(np.random.uniform(0, 1, (size-1, num_split)),dtype = ms.float32) for _ in range(num_samples)]
#             new_data = []
#             for i, sample in enumerate(ori_data):
#                 new_data.append(ops.concat([ops.matmul(matrix_left, sample), ops.matmul(matrix_right, sample), cost_c_data[i]], 1))
#             self.data = new_data
#             self.ori_data = ori_data
#             self.cost_c_data = cost_c_data

#         self.size = len(self.data)

#     def __len__(self):
#         return self.size

#     def __getitem__(self, idx):
#         return self.data[idx], self.ori_data[idx], self.cost_c_data[idx]


# 由于MindSpore框架对于单算子的执行只支持单线程操作，但是在自定义数据集中使用了Tensor的运算操作，即会调到框架的算子执行，由于数据集的处理使用了多线程操作，因此导致整体的执行顺序错乱，出现空指针的错误。
# 将自定义数据集中的Tensor操作改为使用原生numpy进行计算
import numpy as np
import os
import pickle

class PPDataset:
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None, num_split=3):
        super(PPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [np.array(row, dtype=np.float32) for row in (data[offset:offset+num_samples])]
            filename2 = filename.split('/')
            filename2[-1] = 'ori_' + filename2[-1]
            filename2 = '/'.join(filename2)
            filename3 = filename.split('/')
            filename3[-1] = 'cost_c_' + filename3[-1]
            filename3 = '/'.join(filename3)
            with open(filename2, 'rb') as f:
                data = pickle.load(f)   
                self.ori_data = [np.array(row, dtype=np.float32) for row in (data[offset:offset+num_samples])]
            with open(filename3, 'rb') as f:
                data = pickle.load(f)   
                self.cost_c_data = [np.array(row, dtype=np.float32) for row in (data[offset:offset+num_samples])]
        else:
            matrix_left = np.zeros((size-1, size), dtype=np.float32)
            matrix_right = np.zeros((size-1, size), dtype=np.float32)
            for i in range(size-1):
                matrix_left[i, :i+1] = 1.
                matrix_right[i, i+1:] = 1.
            ori_data = [np.random.uniform(0, 1, (size, 1)).astype(np.float32) for _ in range(num_samples)]
            cost_c_data = [np.random.uniform(0, 1, (size-1, num_split)).astype(np.float32) for _ in range(num_samples)]
            new_data = []
            for i, sample in enumerate(ori_data):
                new_data.append(np.concatenate([np.matmul(matrix_left, sample), np.matmul(matrix_right, sample), cost_c_data[i]], 1))
            self.data = new_data
            self.ori_data = ori_data
            self.cost_c_data = cost_c_data
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.ori_data[idx], self.cost_c_data[idx]


# import numpy as np
# import os
# import pickle

# class PPDataset:
#     def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None, num_split=3):
#         super(PPDataset, self).__init__()

#         self.data_set = []
#         if filename is not None:
#             assert os.path.splitext(filename)[1] == '.pkl'

#             with open(filename, 'rb') as f:
#                 data = pickle.load(f)
#                 self.data = [np.array(row, dtype=np.float32) for row in (data[offset:offset+num_samples])]
#             filename2 = filename.split('/')
#             filename2[-1] = 'ori_' + filename2[-1]
#             filename2 = '/'.join(filename2)
#             filename3 = filename.split('/')
#             filename3[-1] = 'cost_c_' + filename3[-1]
#             filename3 = '/'.join(filename3)
#             with open(filename2, 'rb') as f:
#                 data = pickle.load(f)   
#                 self.ori_data = [np.array(row, dtype=np.float32) for row in (data[offset:offset+num_samples])]
#             with open(filename3, 'rb') as f:
#                 data = pickle.load(f)   
#                 self.cost_c_data = [np.array(row, dtype=np.float32) for row in (data[offset:offset+num_samples])]
#         else:
#             matrix_left = np.zeros((size-1, size))
#             matrix_right = np.zeros((size-1, size))
#             for i in range(size-1):
#                 matrix_left[i, :i+1] = 1.
#                 matrix_right[i, i+1:] = 1.
#             ori_data = [np.random.uniform(0, 1, (size, 1)).astype(np.float32) for _ in range(num_samples)]
#             cost_c_data = [np.random.uniform(0, 1, (size-1, num_split)).astype(np.float32) for _ in range(num_samples)]
#             new_data = []
#             for i, sample in enumerate(ori_data):
#                 new_data.append(np.concatenate([np.matmul(matrix_left, sample), np.matmul(matrix_right, sample), cost_c_data[i]], 1))
#             # List:
#             # new_data.shape = [size-1, 2 + num_split],总代价
#             self.data = new_data
#             # ori_data.shape = [size, 1]，原始的layer运行时间,matrix_left/right = [size-1,size] 
#             # 最终：运行时间代价shape = [size-1 , 2]
#             self.ori_data = ori_data
#             # cost_c_data.shape = [size-1, num_split]，原始的通信时间
#             self.cost_c_data = cost_c_data
#         # len(self.data) = num_samples
#         self.size = len(self.data)

#     def __len__(self):
#         return self.size

#     def __getitem__(self, idx):
#         return self.data[idx], self.ori_data[idx], self.cost_c_data[idx]
