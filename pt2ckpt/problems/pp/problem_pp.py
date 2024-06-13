from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.pp.state_pp import StatePP
from utils.beam_search import beam_search
import time

import time
from contextlib import contextmanager
@contextmanager
def timed_block(label):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print(f"{label} : {end - start:.6f}秒")

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

def get_partiton_cost_sequence(data, cost_c_data, partition):
    # print("data:",data,type(data),data.dtype,"cost_c:",cost_c_data,"partition:",partition)
    # time.sleep(0.1)
    # data = torch.Tensor(data)
    pp=len(partition)
    s = partition
    p = [s[0]-1]

    # for i in range(1, pp):
    #     p.append(p[i-1] + s[i])
    # lens = torch.reshape(torch.sum(data[:p[0]+1]), (-1,1))
    # for i in range(len(s)-1):
    #     # print(p[i]+1,p[i+1]+1)
    #     lens = torch.cat([lens,torch.reshape(torch.sum(data[p[i]+1:p[i+1]+1]), (-1,1))])
    # max_sub_seq_cost = lens.view(-1,).max()
    # for i in range(pp-1):
    #     max_sub_seq_cost += cost_c_data[p[i]][i]

    for i in range(1, pp):
        p.append(p[i - 1] + s[i])
    start_time_2 = time.time()
    lens = torch.reshape(torch.sum(data[:p[0]+1]), (-1,1))
    # print(f"Execution time for 2: {time.time() - start_time_2} seconds")

    start_time_3 = time.time()
    for i in range(len(s)-1):
        # print(p[i]+1,p[i+1]+1)
        lens = torch.cat([lens,torch.reshape(torch.sum(data[p[i]+1:p[i+1]+1]), (-1,1))])
    # print(f"Execution time for 3: {time.time() - start_time_3} seconds")

    max_sub_seq_cost = lens.reshape(-1,).max()
    start_time_4 = time.time()
    for i in range(pp-1):
        max_sub_seq_cost += cost_c_data[p[i]][i]
    # print(f"Execution time for 4: {time.time() - start_time_4} seconds")
    
    return max_sub_seq_cost

class PP(object):

    NAME = 'pp'

    @staticmethod
    def get_costs(ori_dataset, cost_c_dataset, dataset, pi):
        node_size = dataset.size(1)+1 #这里输入的node数量是nodesize-1数量
        costs = []
        # start_time_cost = time.time()
        # print("pi:",pi.size(0))
        for idx in range(pi.size(0)):
            position = pi[idx,:]
            position = position.cpu().numpy().tolist()
            data = ori_dataset[idx,:,0]
            cost_data = cost_c_dataset[idx,...]
            # print(data.size())
            partition = pi2partition(position,node_size)
            # print(partition)
            # print(cost_data.size())
            costs.append(get_partiton_cost_sequence(data,cost_data,partition))
            # print(data.size())
        # print(f"Execution time for cosst: {time.time() - start_time_cost} seconds")
        costs = torch.Tensor(costs)[:,None].to(dataset.device)
        return costs,None 

    @staticmethod
    def make_dataset(*args, **kwargs):
        return PPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StatePP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = PP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class PPDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None,num_split=3):
        super(PPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
            filename2 = filename.split('/')
            filename2[-1] = 'ori_' + filename2[-1]
            filename2 = '/'.join(filename2)
            filename3 = filename.split('/')
            filename3[-1] = 'cost_c_' + filename3[-1]
            filename3 = '/'.join(filename3)
            with open(filename2, 'rb') as f:
                data = pickle.load(f)   
                self.ori_data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
            with open(filename3, 'rb') as f:
                data = pickle.load(f)   
                self.cost_c_data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
        else:
            # Sample points randomly in [0, 1] square
            # ori_data = [torch.FloatTensor(size, 1).uniform_(0, 1) for i in range(num_samples)]
            # new_data = []
            # for i, sample in enumerate(ori_data):
            #     new_sample = []
            #     for j in range(sample.size(0)-1):
            #         new_sample.append([sample[:j+1,:].sum(),sample[j+1:,:].sum()])
            #     new_data.append(torch.Tensor(new_sample))
            # self.data=new_data
            # self.ori_data = ori_data
            # 老代码太慢，升级如下
            # ori_data = torch.FloatTensor(num_samples, size, 1).uniform_(0, 1) #for i in range(num_samples)
            # matrix_left = torch.zeros((size-1,size))
            # matrix_right = torch.zeros((size-1,size))
            # for i in range(size-1):
            #     matrix_left[i,:i+1]=1.
            #     matrix_right[i,i+1:]=1.
            # left_data = torch.bmm(matrix_left[None,...].expand(num_samples,size-1,size),ori_data)
            # right_data = torch.bmm(matrix_right[None,...].expand(num_samples,size-1,size),ori_data)
            # data = torch.cat([left_data,right_data],2)
            # self.data = data
            # self.ori_data = ori_data
            # V3.0 升级
            matrix_left = torch.zeros((size-1,size))
            matrix_right = torch.zeros((size-1,size))
            for i in range(size-1):
                matrix_left[i,:i+1]=1.
                matrix_right[i,i+1:]=1.
            ori_data = [torch.FloatTensor(size, 1).uniform_(0, 1) for i in range(num_samples)]
            cost_c_data = [torch.FloatTensor(size-1, num_split).uniform_(0, 1) for i in range(num_samples)]
            new_data = []
            for i, sample in enumerate(ori_data):
                new_data.append(torch.cat([torch.mm(matrix_left,sample),torch.mm(matrix_right,sample),cost_c_data[i]],1))
            self.data=new_data
            self.ori_data = ori_data
            self.cost_c_data = cost_c_data
            # v 4.0 升级
            # self.matrix_left = torch.zeros((size-1,size))
            # self.matrix_right = torch.zeros((size-1,size))
            # for i in range(size-1):
            #     self.matrix_left[i,:i+1]=1.
            #     self.matrix_right[i,i+1:]=1.
            

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
#         num_samples = 1
#         ori_data = [torch.FloatTensor(size, 1).uniform_(0, 1) for i in range(num_samples)]
#         cost_c_data = [torch.FloatTensor(size-1, num_split).uniform_(0, 1) for i in range(num_samples)]
#         new_data = []
#         for i, sample in enumerate(ori_data):
#             new_data.append(torch.cat([torch.mm(self.matrix_left,sample),torch.mm(self.matrix_right,sample),cost_c_data[i]],1))
#         new_data
#         ori_data
#         cost_c_data
#         return data[0], ori_data[0], cost_c_data[0]
        
        
        
        
        return self.data[idx], self.ori_data[idx], self.cost_c_data[idx] 
