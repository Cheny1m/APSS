import os
import pickle
import numpy as np
import psutil
import os

from mindspore import Tensor
import mindspore as ms
import mindspore.ops as ops

from apss.utils.beam_search import beam_search

from.state_pp import StatePP # ,initialize_pp_state


# DPSN的数据生成
class PP(object):
    NAME = 'pp'

    # 计算并返回成本
    @staticmethod
    def get_costs(ori_dataset, cost_c_dataset, dataset, pi):
        # 原始方法：
        # node_size = dataset.shape[1] + 1  # 这里输入的node数量是nodesize-1数量
        # costs = []
        # for idx in range(pi.shape[0]):
        #     position = pi[idx, :]
        #     position = position.asnumpy().tolist()
        #     data = ori_dataset[idx, :, 0]
        #     cost_data = cost_c_dataset[idx, ...]
        #     partition = pi2partition(position, node_size) 
        #     costs.append(get_partition_cost_sequence(data, cost_data, partition))
            
        # # costs_np= np.array([item.asnumpy() for item in costs])
        # # costs_tensor = Tensor(costs_np)[:, None]
        # # return costs_tensor, None
            
        # costs = ops.stack(costs)[:,None]
        # return costs, None

        # 快速方法：
        node_size = dataset.shape[1] + 1  # 这里输入的node数量是nodesize-1数量
        pi,_ = ops.sort(pi,-1)
        pi =ops.cast(pi,ms.int32)
        node_size_tensor = ops.full((pi.shape[0],1),node_size-1,dtype=ms.int32)
        p = ops.concat((pi,node_size_tensor),-1)
        data = ops.cumsum(ori_dataset.squeeze(-1),-1)
        index_data = ops.gather(data,p,1,1)
        # 在nvidia环境中请使用下面两行代码：
        # index_data = ops.cast(index_data,ms.int32)
        # index_data = ops.concat((ops.zeros((index_data.shape[0],1),dtype=ms.int32),index_data),-1)
        # 在mindspore环境中，请使用这两行代码：
        index_data = ops.cast(index_data,ms.float32)
        index_data = ops.concat((ops.zeros((index_data.shape[0],1)),index_data),-1)
        max_sub_seq_cost,_  = ops.max(ops.diff(index_data),-1)
        indices = ops.tile(ops.arange(pi.shape[-1]),(cost_c_dataset.shape[0],1))
        cost_data = cost_c_dataset.reshape(cost_c_dataset.shape[0],cost_c_dataset.shape[1]*cost_c_dataset.shape[2])
        indices1 = pi*pi.shape[-1] + indices
        cost_cost = ops.sum(ops.gather_elements(cost_data, 1,indices1),-1)
        max_sub_seq_cost += cost_cost
        return max_sub_seq_cost,None

    # 返回一个PPDataset实例
    @staticmethod
    def make_dataset(*args, **kwargs):
        return PPDataset(*args, **kwargs)

    # make_state: 初始化并返回一个StatePP实例。
    @staticmethod
    def make_state(*args, **kwargs):
        return StatePP.initialize(*args, **kwargs)
        # return initialize_pp_state(*args, **kwargs)

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
# mindspore通过每次调用Python层自定义的Dataset以生成数据集，而Pytorch自定义数据集的抽象类，然后进行继承

# 由于MindSpore框架对于单算子的执行只支持单线程操作，但是在自定义数据集中使用了Tensor的运算操作，即会调到框架的算子执行，由于数据集的处理使用了多线程操作，因此导致整体的执行顺序错乱，出现空指针的错误。
# 将自定义数据集中的Tensor操作改为使用原生numpy进行计算,从而保证可以多线程执行
#     

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
