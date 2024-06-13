from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.sp.state_sp import StateSP
from utils.beam_search import beam_search


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

def get_partiton_cost_sequence(data,partition):
    # data = torch.Tensor(data)
    pp=len(partition)
    s = partition
    p = [s[0]-1]

    for i in range(1, pp):
        p.append(p[i-1] + s[i])
    lens = torch.reshape(torch.sum(data[:p[0]+1]), (-1,1))
    for i in range(len(s)-1):
        # print(p[i]+1,p[i+1]+1)
        lens = torch.cat([lens,torch.reshape(torch.sum(data[p[i]+1:p[i+1]+1]), (-1,1))])
    return lens.view(-1,)

class SP(object):

    NAME = 'sp'

    @staticmethod
    def get_costs(ori_dataset, dataset, pi):
        node_size = dataset.size(1)+1 #这里输入的node数量是nodesize-1数量
        costs = []
        for idx in range(pi.size(0)):
            position = pi[idx,:].cpu().numpy().tolist()
            
            data = ori_dataset[idx,:,0]
            # print(data.size())
            partition = pi2partition(position,node_size)
            # print(partition)
            costs.append(get_partiton_cost_sequence(data,partition).max())
            # print(data.size())
        costs = torch.Tensor(costs)[:,None].to(dataset.device)
        return costs,None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return SPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateSP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = SP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class SPDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(SPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
            filename2 = filename.split('/')
            filename2[-1] = 'ori_' + filename2[-1]
            filename2 = '/'.join(filename2)
            with open(filename2, 'rb') as f:
                data = pickle.load(f)   
                self.ori_data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
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
            new_data = []
            for i, sample in enumerate(ori_data):
                new_data.append(torch.cat([torch.mm(matrix_left,sample),torch.mm(matrix_right,sample)],1))
            self.data=new_data
            self.ori_data = ori_data

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.ori_data[idx] 
