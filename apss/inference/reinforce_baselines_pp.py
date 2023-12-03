import copy
from scipy.stats import ttest_rel

import mindspore
from mindspore import nn

from .train_mc import rollout

class Baseline(object):

    def wrap_dataset(self, dataset):
        return dataset

    def unwrap_batch(self, batch):
        return batch, None, None
        # return batch['data'], batch['baseline_cost'].view(-1), batch['baseline_pi']

    def eval(self, x, c):
        raise NotImplementedError("Override this method")

    def get_learnable_parameters(self):
        return []

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

class WarmupBaseline(nn.Cell):

    def __init__(self, baseline, n_epochs=1, warmup_exp_beta=0.8):
        super(WarmupBaseline, self).__init__()

        self.baseline = baseline
        assert n_epochs > 0, "n_epochs to warmup must be positive"
        self.warmup_baseline = ExponentialBaseline(warmup_exp_beta)
        self.alpha = 0
        self.n_epochs = n_epochs

    def wrap_dataset(self, dataset):
        if self.alpha > 0:
            return self.baseline.wrap_dataset(dataset)
        return self.warmup_baseline.wrap_dataset(dataset)

    def unwrap_batch(self, batch):
        if self.alpha > 0:
            return self.baseline.unwrap_batch(batch)
        return self.warmup_baseline.unwrap_batch(batch)

    def eval(self, split_x, ori_x, c):
        if self.alpha == 1:
            return self.baseline.eval(split_x, ori_x, c)
        if self.alpha == 0:
            return self.warmup_baseline.eval(split_x, c)
        v, l = self.baseline.eval(split_x, ori_x, c)
        vw, lw = self.warmup_baseline.eval(split_x, c)
        # Return convex combination of baseline and of loss
        return self.alpha * v + (1 - self.alpha) * vw, self.alpha * l + (1 - self.alpha * lw)

    def epoch_callback(self, model, epoch):
        # Need to call epoch callback of inner model (also after first epoch if we have not used it)
        self.baseline.epoch_callback(model, epoch)
        self.alpha = (epoch + 1) / float(self.n_epochs)
        if epoch < self.n_epochs:
            print("Set warmup alpha = {}".format(self.alpha))

    def construct(self, split_x, ori_x, c):
        if self.alpha == 1:
            return self.baseline.construct(split_x, ori_x, c)
        if self.alpha == 0:
            return self.warmup_baseline.construct(split_x, c)
        v, l = self.baseline.construct(split_x, ori_x, c)
        vw, lw = self.warmup_baseline.construct(split_x, c)
        # Return convex combination of baseline and of loss
        return self.alpha * v + (1 - self.alpha) * vw, self.alpha * l + (1 - self.alpha * lw)

    def state_dict(self):
        # Checkpointing within warmup stage makes no sense, only save inner baseline
        return self.baseline.state_dict()

    def load_state_dict(self, state_dict):
        # Checkpointing within warmup stage makes no sense, only load inner baseline
        self.baseline.load_state_dict(state_dict)


class NoBaseline(Baseline):

    def eval(self, x, c):
        return 0, 0  # No baseline, no loss


class ExponentialBaseline(Baseline):

    def __init__(self, beta):
        super(Baseline, self).__init__()

        self.beta = beta
        self.v = None

    def eval(self, x, c):

        if self.v is None:
            v = c.mean()
        else:
            v = self.beta * self.v + (1. - self.beta) * c.mean()

        self.v = v.detach()  # Detach since we never want to backprop
        return self.v, 0  # No loss

    def state_dict(self):
        return {
            'v': self.v
        }

    def load_state_dict(self, state_dict):
        self.v = state_dict['v']


class CriticBaseline(Baseline):

    def __init__(self, critic):
        super(Baseline, self).__init__()

        self.critic = critic

    def eval(self, x, c):
        v = self.critic(x)
        # Detach v since actor should not backprop through baseline, only for loss
        return v.detach(), nn.MSELoss(v, c.detach())

    def get_learnable_parameters(self):
        return list(self.critic.parameters())

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return {
            'critic': self.critic.state_dict()
        }

    def load_state_dict(self, state_dict):
        critic_state_dict = state_dict.get('critic', {})
        if not isinstance(critic_state_dict, dict):  # backwards compatibility
            critic_state_dict = critic_state_dict.state_dict()
        self.critic.load_state_dict({**self.critic.state_dict(), **critic_state_dict})

class RolloutBaselinePP(Baseline):
    def __init__(self, model, problem, opts, epoch=0):
        super(RolloutBaselinePP, self).__init__()

        self.problem = problem
        self.opts = opts

        #Setting up or updating models and associated datasets
        self._update_model(model, epoch)
        # self.alpha = 0.

    def _update_model(self, model, epoch, dataset=None):
        self.model = copy.deepcopy(model)
        # Always generate baseline dataset when updating model to prevent overfitting to the baseline dataset

        if dataset is not None:
            if len(dataset) != self.opts.val_size:
                print("Warning: not using saved baseline dataset since val_size does not match")
                dataset = None
            elif self.problem.NAME == 'tsp' and dataset[0].shape[0] != self.opts.graph_size:
                print("Warning: not using saved baseline dataset since graph_size does not match")
                dataset = None
            elif self.problem.NAME in ['sp', 'pp'] and dataset[0][1].shape[0] != self.opts.graph_size:
                print("Warning: not using saved baseline dataset since graph_size does not match")
                dataset = None

        if dataset is None:
            self.dataset = self.problem.make_dataset(size=self.opts.graph_size, num_samples=self.opts.val_size, distribution=self.opts.data_distribution, num_split=self.opts.num_split)
        else:
            self.dataset = dataset
        # Type: list
        # number = val_size,
        # Each sample has shape [size-1, 2 + num_split]
        # print("len(dataset)",len(self.dataset))
        # print(self.dataset)
        print("Evaluating baseline model on evaluation dataset")
        self.bl_vals, _ = rollout(self.model, self.dataset, self.opts)
        self.bl_vals = self.bl_vals.asnumpy()
        self.mean = self.bl_vals.mean()
        self.epoch = epoch

    def wrap_dataset(self, dataset):
        print("Evaluating baseline on dataset...")
        # Need to convert baseline to 2D to prevent converting to double
        baseline_output = rollout(self.model, dataset, self.opts)
        # baseline_cost = baseline_output[0].astype(np.float32).reshape(-1, 1)
        # baseline_pi = baseline_output[1].astype(np.float32)
        # return BaselineDataset(dataset, (baseline_cost, baseline_pi))
        return BaselineDataset(dataset, baseline_output)

    def unwrap_batch(self, batch):
        return batch['data'], batch['baseline_cost'].reshape(-1), batch['baseline_pi']

    def eval(self, split_x, ori_x, cost_c_x):
        print("use eval!")
        # Use no grad for efficient inference
        # mindspore.context.set_context(mode=mindspore.context.GRAPH_MODE)
        mindspore.context.set_context(mode=mindspore.PYNATIVE_MODE)
        
        with mindspore.no_grad():
            v, _ = self.model(split_x, ori_x, cost_c_x, return_pi=False)
        return v, 0

    def epoch_callback(self, model, epoch):
        print("Evaluating candidate model on evaluation dataset")
        candidate_vals, candidate_pis = rollout(model, self.dataset, self.opts)
        candidate_vals = candidate_vals.asnumpy()
        candidate_mean = candidate_vals.mean()
        print("Epoch {} candidate mean {}, baseline epoch {} mean {}, difference {}".format(
            epoch, candidate_mean, self.epoch, self.mean, candidate_mean - self.mean))
        if candidate_mean - self.mean < 0:
            # Calc p value
            t, p = ttest_rel(candidate_vals, self.bl_vals)

            p_val = p / 2  # one-sided
            assert t < 0, "T-statistic should be negative"
            print("p-value: {}".format(p_val))
            if p_val < self.opts.bl_alpha:
                print('Update baseline')
                self._update_model(model, epoch)

    def state_dict(self):
        return {
            'model': self.model,
            'dataset': self.dataset,
            'epoch': self.epoch
        }

    def load_state_dict(self, state_dict):
        load_model = copy.deepcopy(self.model)
        load_model.set_train(False)
        self._update_model(load_model, state_dict['epoch'], state_dict['dataset'])

    
class BaselineDataset:

    def __init__(self, dataset=None, baseline=None):
        super(BaselineDataset, self).__init__()

        self.dataset = dataset
        self.baseline_cost = baseline[0].reshape(-1, 1).asnumpy()
        self.baseline_pi = baseline[1]
        # print("self.dataset",len(self.dataset),"self.baseline_cost",len(self.baseline_cost),"self.baseline_pi",len(self.baseline_pi))
        assert len(self.dataset) == len(self.baseline_cost), print(len(self.dataset), len(self.baseline_cost))

    def __getitem__(self, item):
        item = int(item)
        # return {
        #     'data': self.dataset[item],
        #     'baseline_cost': self.baseline_cost[item],
        #     'baseline_pi': self.baseline_pi[item, :]
        # }
        # 返回列表，而不是字典
        return self.dataset[item], self.baseline_cost[item], self.baseline_pi[item, :]


    def __len__(self):
        return len(self.dataset)