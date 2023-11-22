import numpy as np

from mindspore import nn
from mindspore import Tensor
from mindspore.ops import functional as F

class CustomReinforceLoss(nn.Cell):
    # def __init__(self, model, baseline):
    #     super(CustomReinforceLoss, self).__init__()
    #     self.model = model
    #     self.baseline = baseline
    def __init__(self):
        super(CustomReinforceLoss, self).__init__()
        self.reinforce_loss = None
        self.bl_loss = None

    def construct(self, cost, cost2, bl_cost, log_likelihood, log_likelihood2):
        # 样本级别对比奖励
        loss_weight = Tensor(np.ones(cost.shape[0]), dtype=cost.dtype)
        loss_weight[cost.reshape(-1) < cost2.reshape(-1)] = -1.
        loss_weight[cost.reshape(-1) == cost2.reshape(-1)] = 0.

        loss_weight2 = Tensor(np.ones(cost.shape[0]), dtype=cost.dtype)
        loss_weight2[cost.reshape(-1) > cost2.reshape(-1)] = -1.
        loss_weight2[cost.reshape(-1) == cost2.reshape(-1)] = 0.

        # 与bl相比较
        loss_weight_bl = Tensor(np.ones(cost.shape[0]), dtype=cost.dtype)
        loss_weight_bl[cost.reshape(-1) < bl_cost.reshape(-1)] = -1.
        loss_weight_bl[cost.reshape(-1) == bl_cost.reshape(-1)] = 0.

        loss_weight_bl2 = Tensor(np.ones(cost.shape[0]), dtype=cost.dtype)
        loss_weight_bl2[cost2.reshape(-1) < bl_cost.reshape(-1)] = -1.
        loss_weight_bl2[cost2.reshape(-1) == bl_cost.reshape(-1)] = 0.
        
 
        self.reinforce_loss = ((loss_weight + loss_weight_bl) * log_likelihood).mean() + ((loss_weight2 + loss_weight_bl2) * log_likelihood2).mean()
        self.bl_loss = 0
        loss = self.reinforce_loss + self.bl_loss
        return loss
    
    # @staticmethod
    def get_loss(self):
        return self.reinforce_loss, self.bl_loss


# # use contrastive loss
# class CustomReinforceLoss(nn.Cell):
#     # def __init__(self, model, baseline):
#     #     super(CustomReinforceLoss, self).__init__()
#     #     self.model = model
#     #     self.baseline = baseline
#     def __init__(self,model):
#         super(CustomReinforceLoss, self).__init__()
#         self.model = model
#         self.reinforce_loss = None
#         self.bl_loss = None

#     def construct(self, split_x, ori_x, cost_c_x, return_pi=True):

#         cost, log_likelihood, pi = self.model(split_x, ori_x, cost_c_x, return_pi=True)
#         cost2, log_likelihood2, pi2 = self.model(split_x, ori_x, cost_c_x, return_pi=True)
#         # 样本级别对比奖励
#         loss_weight = Tensor(np.ones(cost.shape[0]), dtype=cost.dtype)
#         loss_weight[cost.reshape(-1) < cost2.reshape(-1)] = -1.
#         loss_weight[cost.reshape(-1) == cost2.reshape(-1)] = 0.

#         loss_weight2 = Tensor(np.ones(cost.shape[0]), dtype=cost.dtype)
#         loss_weight2[cost.reshape(-1) > cost2.reshape(-1)] = -1.
#         loss_weight2[cost.reshape(-1) == cost2.reshape(-1)] = 0.

#         # 与bl相比较
#         loss_weight_bl = Tensor(np.ones(cost.shape[0]), dtype=cost.dtype)
#         loss_weight_bl[cost.reshape(-1) < bl_cost.reshape(-1)] = -1.
#         loss_weight_bl[cost.reshape(-1) == bl_cost.reshape(-1)] = 0.

#         loss_weight_bl2 = Tensor(np.ones(cost.shape[0]), dtype=cost.dtype)
#         loss_weight_bl2[cost2.reshape(-1) < bl_cost.reshape(-1)] = -1.
#         loss_weight_bl2[cost2.reshape(-1) == bl_cost.reshape(-1)] = 0.
        
 
#         self.reinforce_loss = ((loss_weight + loss_weight_bl) * log_likelihood).mean() + ((loss_weight2 + loss_weight_bl2) * log_likelihood2).mean()
#         self.bl_loss = 0
#         loss = self.reinforce_loss + self.bl_loss
#         return loss
    
#     @staticmethod
#     def get_loss(self):
#         return self.reinforce_loss, self.bl_loss

    

    