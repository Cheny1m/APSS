import math
import psutil
import os
import json

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype
import mindspore.ops as ops
from mindspore.common.initializer import initializer, Uniform


from apss.utils.tensor_functions import compute_in_batches
from apss.utils.beam_search import CachedLookup
from apss.utils.functions import sample_many

from .graph_encoder import GraphAttentionEncoder



def set_decode_type(model, decode_type):
    # if isinstance(model, data_parallel):
    # model只是实例化了一个Cell类，没有调用Model类进行封装
    # model = model._network
    model.set_decode_type(decode_type)

class AttentionModelFixed:
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    def __init__(self, 
                 node_embeddings: ms.Tensor, 
                 context_node_projected: ms.Tensor,
                 glimpse_key: ms.Tensor,
                 glimpse_val: ms.Tensor,
                 logit_key: ms.Tensor):
        self.node_embeddings = node_embeddings
        self.context_node_projected = context_node_projected
        self.glimpse_key = glimpse_key
        self.glimpse_val = glimpse_val
        self.logit_key = logit_key

    def __getitem__(self, key):
        assert isinstance(key, (ms.Tensor, slice))
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
            glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
            logit_key=self.logit_key[key]
        )

    def __repr__(self):
        return f"AttentionModelFixed(node_embeddings={self.node_embeddings}, context_node_projected={self.context_node_projected}, glimpse_key={self.glimpse_key}, glimpse_val={self.glimpse_val}, logit_key={self.logit_key})"

class AttentionStepModel(nn.Cell):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None,
                 num_split=None,
                 node_size=None):
        super(AttentionStepModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.allow_partial = problem.NAME == 'sdvrp'
        self.is_vrp = problem.NAME == 'cvrp' or problem.NAME == 'sdvrp'
        self.is_orienteering = problem.NAME == 'op'
        self.is_pctsp = problem.NAME == 'pctsp'
        self.is_sp = problem.NAME == 'sp'
        self.is_pp = problem.NAME == 'pp'

        self.node_size = node_size 
        self.num_split = num_split
        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size

        step_context_dim = embedding_dim
        node_dim = 2 + self.num_split
        self.W_placeholder = ms.Parameter(initializer(Uniform(scale=1), [embedding_dim],ms.float32))
        
        self.init_embed = nn.Dense(node_dim, embedding_dim)

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )
        self.project_node_embeddings = nn.Dense(embedding_dim, 3 * embedding_dim,has_bias = False)
        self.project_fixed_context = nn.Dense(embedding_dim, embedding_dim,has_bias = False)
        self.project_step_context = nn.Dense(step_context_dim, embedding_dim,has_bias = False)
        self.project_selected_context = nn.Dense(self.node_size-1, embedding_dim, has_bias=False)
        self.project_remaining_context = nn.Dense(1, embedding_dim, has_bias=False)
        
        assert embedding_dim % n_heads == 0
        self.project_out = nn.Dense(embedding_dim, embedding_dim,has_bias = False)
 
        # self._multinomial = ops.Multinomial()

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:
            self.temp = temp

    # @ms.jit
    # @profile
    def construct(self, input, ori_input, cost_c_input, return_pi=False):

        embeddings, _ = self.embedder(self.init_embed(input))
        # decoder、Context embedding、Calculation of log-probabilities
        _log_p, pi = self._inner(input, embeddings) 
        cost, mask = self.problem.get_costs(ori_input, cost_c_input, input, pi) 
        ll = self._calc_log_likelihood(_log_p, pi, mask)
    
        if return_pi:
            return cost, ll, pi
        return cost, ll



    def beam_search(self, *args, **kwargs):
        return self.problem.beam_search(*args, **kwargs, model=self)

    def precompute_fixed(self, input):
        embeddings, _ = self.embedder(self._init_embed(input))
        return CachedLookup(self._precompute(embeddings))

    def propose_expansions(self, beam, fixed, expand_size=None, normalize=False, max_calc_batch_size=4096):
        log_p_topk, ind_topk = compute_in_batches(
            lambda b: self._get_log_p_topk(fixed[b.ids], b.state, k=expand_size, normalize=normalize),
            max_calc_batch_size, beam, n=beam.size()
        )

        assert log_p_topk.shape[1] == 1, "Can only have single step"
        score_expand = beam.score[:, None] + log_p_topk[:, 0, :]

        flat_action = ind_topk.reshape(-1)
        flat_score = score_expand.reshape(-1)
        flat_feas = flat_score > -1e10

        flat_parent = ops.arange(flat_action.shape[1], dtype=mstype.int64, device=flat_action.device) // ind_topk.shape[1]
        feas_ind_2d = ops.nonzero(flat_feas)

        if len(feas_ind_2d) == 0:
            return None, None, None

        feas_ind = feas_ind_2d[:, 0]

        return flat_parent[feas_ind], flat_action[feas_ind], flat_score[feas_ind]

    def _calc_log_likelihood(self, _log_p, a, mask):
        log_p = ops.gather_elements(_log_p, 2, ops.expand_dims(a, -1)).squeeze(-1)

        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).all(), "Logprobs should not be -inf, check sampling procedure!"

        return log_p.sum(1)

    def _init_embed(self, input):
        return self.init_embed(input)
    
    # @profile
    def _inner(self, input, embeddings):
        outputs = []
        sequences = []
        state = self.problem.make_state(input)
     
        fixed = self._precompute(embeddings)
        i = 0
        selected_dividers = ops.zeros((embeddings.shape[0],1,embeddings.shape[-1]))
        while True:
            if i >= self.num_split:
                break
            log_p, mask = self._get_log_p(fixed, state,selected_dividers) # 传入两个实例StatePP和AttentionModelFixed
            # selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])
            selected = self._select_node(ops.squeeze(ops.exp(log_p),1), ops.squeeze(mask,1))
            state = state.update(selected)

            outputs.append(ops.squeeze(log_p,1))
            sequences.append(selected)

            i += 1

        return ops.stack(outputs, 1), ops.stack(sequences, 1)

    def sample_many(self, input, batch_rep=1, iter_rep=1):
        return sample_many(
            lambda input: self._inner(*input),
            lambda input, pi: self.problem.get_costs(input[0], pi),
            (input, self.embedder(self._init_embed(input))[0]),
            batch_rep, iter_rep
        )

    # @profile
    # @ms.jit
    def _select_node(self, probs, mask):
        assert (probs == probs).all(), "Probs should not contain any nans"
        if self.decode_type == "greedy":
            selected = probs.argmax(1)
            assert not ops.gather_elements(mask,1, ops.expand_dims(selected,-1)).any(), "Decode greedy: infeasible action has maximum probability"
        
        elif self.decode_type == "sampling":
            selected = ops.multinomial(probs, 1,replacement = False).squeeze(1)
            #  for https://gitee.com/mindspore/mindspore/issues/I5NEFI?from=project-issue,but got RuntimeError: Illegal primitive Multinomial's bprop not defined
            # selected = self._multinomial(probs, 1).squeeze(1)
            # while ops.gather_elements(mask,1,selected.unsqueeze(-1)).any():
            while ops.ReduceAny()(ops.gather_elements(mask,1,selected.unsqueeze(-1))):
                print('Sampled bad values, resampling!')
                selected = ops.multinomial(probs, 1,replacement = False).squeeze(1) 
                # selected = self._multinomial(probs, 1).squeeze(1)   
        else:
            assert False, "Unknown decode type"
        return selected

    def _precompute(self, embeddings, num_steps=1):
        graph_embed = embeddings.mean(1) # 在第二个维度求mean，graph_embed.shape = [1000,128]
        fixed_context = ops.expand_dims(self.project_fixed_context(graph_embed),1)
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = self.project_node_embeddings(ops.expand_dims(embeddings,1)).chunk(3, axis=-1)
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps), # shape = [8,1000,1,7,16]
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
            # logit_key_fixed
        ) # 包含三个Tensor对象的元组 
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _get_log_p_topk(self, fixed, state, k=None, normalize=True):
        log_p, _ = self._get_log_p(fixed, state, normalize=normalize)

        if k is not None and k < log_p.shape[-1]:
            return log_p.topk(k, -1)

        return log_p, ops.arange(log_p.shape[-1], dtype=ops.dtype.int64).repeat(log_p.shape[0], 1)[:, None, :]

    # @profile
    def _get_log_p(self, fixed, state, selected_dividers,normalize=True):
        step_embedding = self._get_parallel_step_context(fixed.node_embeddings, state)
        selected_dividers += self.project_step_context(step_embedding)
        selected_states = self.project_selected_context(state.visited.float())
        remain_steps = self.project_remaining_context((self.num_split - state.visited.sum(axis=-1,keepdims=True)).float())
        query = fixed.context_node_projected + selected_dividers + selected_states + remain_steps
        glimpse_K, glimpse_V, logit_K = fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

        mask = state.get_mask() # mask.shape = [1000,1,7]

        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)

        if normalize:
            log_p = ops.log_softmax(log_p / self.temp, axis=-1)

        assert not ops.isnan(log_p).any()

        return log_p, mask


    def _get_parallel_step_context(self, embeddings, state, from_depot=False):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)
    
        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """

        current_node = state.get_current_node()  # batchsize,num_steps = [1000,1]
        # batch_size, num_steps = current_node.shape
        batch_size = current_node.shape[0]
        num_steps = current_node.shape[1]
        
        if num_steps == 1:
            if state.i.item() == 0:
                # return ops.broadcast_to(self.W_placeholder[None, None, :], (batch_size, 1, self.W_placeholder.shape[-1]))
                return ops.tile(self.W_placeholder, (batch_size, 1, 1))
            else:
                # return ops.gather_elements(embeddings,1,ops.broadcast_to(current_node[:,:,None], (batch_size , 2 , embeddings.shape[-1]))).reshape(batch_size, 1, -1)
                return ops.gather_elements(embeddings,1,ops.tile(current_node[:,:,None], (1 , 1 , embeddings.shape[-1]))).reshape(batch_size, 1, -1)
        else:
            print("ERROR in _get_parallel_step_context")    

   
    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):
        batch_size, num_steps, embed_dim = query.shape
        key_size = val_size = embed_dim // self.n_heads

        glimpse_Q = ops.transpose(query.reshape(batch_size, num_steps, self.n_heads, 1, key_size),(2, 0 , 1, 3, 4))

        compatibility = ops.matmul(glimpse_Q, glimpse_K.transpose(0, 1, 2, 4, 3)) / math.sqrt(glimpse_Q.shape[-1])
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"

            # 替换切片操作
            # mask = ops.expand_dims((ops.expand_dims(mask,0)),3).expand_as(compatibility)
            compatibility = ops.masked_fill(compatibility,ops.expand_dims((ops.expand_dims(mask,0)),3),-math.inf)
        
        # print("glimpse_V",glimpse_V,glimpse_V.shape)
        heads = ops.matmul(ops.softmax(compatibility, axis=-1), glimpse_V)

        glimpse = self.project_out(
            heads.transpose(1, 2, 3, 0, 4).reshape(-1, num_steps, 1, self.n_heads * val_size)
        )

        final_Q = glimpse
        # print("logit_K.shape",logit_K.shape)
        logits = ops.matmul(final_Q, logit_K.transpose(0, 1, 3, 2)).squeeze(-2) / math.sqrt(final_Q.shape[-1])

        if self.tanh_clipping > 0:
            logits = ops.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits = ops.masked_fill(logits,mask,-math.inf)

        return logits, glimpse.squeeze(-2)

    def _get_attention_node_data(self, fixed, state):

        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.shape[1] == 1 or v.shape[1] == num_steps

        return (
            v.view((v.shape[0], v.shape[1], v.shape[2], self.n_heads, -1))
            .broadcast_to((v.shape[0], v.shape[1] if num_steps is None else num_steps, v.shape[2], self.n_heads, -1))
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )
        
