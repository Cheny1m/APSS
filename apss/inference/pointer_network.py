# import torch
# import torch.nn as nn
# from torch.autograd import Variable
import math
import numpy as np

import math
import numpy as np
import mindspore.nn as nn
from mindspore.ops import operations as P
import mindspore.ops as ops
from mindspore import Tensor, Parameter
from mindspore.common import initializer as init
import mindspore.common.initializer as initializer

class Encoder(nn.Cell):
    """Maps a graph represented as an input sequence to a hidden vector"""

    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.init_hx, self.init_cx = self.init_hidden(hidden_dim)

    def construct(self, x, hidden):
        output, hidden = self.lstm(x, hidden)
        return output, hidden

    def init_hidden(self, hidden_dim):
        """Trainable initial hidden state"""
        std = 1. / math.sqrt(hidden_dim)
        enc_init_hx = Parameter(init.uniform_(Tensor(hidden_dim), -std, std))
        enc_init_cx = Parameter(init.uniform_(Tensor(hidden_dim), -std, std))
        return enc_init_hx, enc_init_cx


class Attention(nn.Cell):
    """A generic attention module for a decoder in seq2seq"""
    def __init__(self, dim, use_tanh=False, C=10):
        super(Attention, self).__init__()
        self.use_tanh = use_tanh
        self.project_query = nn.Dense(dim, dim)
        self.project_ref = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.C = C  # tanh exploration
        self.tanh = ops.Tanh()

        self.v = Parameter(Tensor(dim))
        init.initializer.Uniform(-(1. / math.sqrt(dim)), 1. / math.sqrt(dim))(self.v)
        
    def construct(self, query, ref):
        """
        Args: 
            query: is the hidden state of the decoder at the current
                time step. batch x dim
            ref: the set of hidden states from the encoder. 
                sourceL x batch x hidden_dim
        """
        # ref is now [batch_size x hidden_dim x sourceL]
        ref = ref.transpose(1, 2)
        q = self.project_query(query).unsqueeze(2)  # batch x dim x 1
        e = self.project_ref(ref)  # batch_size x hidden_dim x sourceL 
        # expand the query by sourceL
        # batch x dim x sourceL
        expanded_q = ops.Tile()(q, (1, 1, e.shape[2])) 
        # batch x 1 x hidden_dim
        v_view = ops.ExpandDims()(self.v, 1).expand(
                expanded_q.shape[0], len(self.v)).unsqueeze(1)
        # [batch_size x 1 x hidden_dim] * [batch_size x hidden_dim x sourceL]
        u = ops.squeeze(ops.BatchMatMul()(v_view, self.tanh(expanded_q + e)), 1)
        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u  
        return e, logits

class Decoder(nn.Cell):
    def __init__(self, 
            embedding_dim,
            hidden_dim,
            tanh_exploration,
            use_tanh,
            n_glimpses=1,
            mask_glimpses=True,
            mask_logits=True):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_glimpses = n_glimpses
        self.mask_glimpses = mask_glimpses
        self.mask_logits = mask_logits
        self.use_tanh = use_tanh
        self.tanh_exploration = tanh_exploration
        self.decode_type = None  # Needs to be set explicitly before use

        self.lstm = nn.LSTMCell(embedding_dim, hidden_dim)
        self.pointer = Attention(hidden_dim, use_tanh=use_tanh, C=tanh_exploration)
        self.glimpse = Attention(hidden_dim, use_tanh=False)
        self.sm = nn.Softmax(axis=1)

    def update_mask(self, mask, selected):
        return mask.clone().scatter_(1, selected.unsqueeze(-1), True)

    def recurrence(self, x, h_in, prev_mask, prev_idxs, step, context):

        logit_mask = self.update_mask(prev_mask, prev_idxs) if prev_idxs is not None else prev_mask

        logits, h_out = self.calc_logits(x, h_in, logit_mask, context, self.mask_glimpses, self.mask_logits)

        # Calculate log_softmax for better numerical stability
        log_p = ops.LogSoftmax(axis=1)(logits)
        probs = ops.Exp(log_p)

        if not self.mask_logits:
            probs = ops.tensorops.where(logit_mask, ops.ZerosLike()(probs), probs)

        return h_out, log_p, probs, logit_mask

    def calc_logits(self, x, h_in, logit_mask, context, mask_glimpses=None, mask_logits=None):

        if mask_glimpses is None:
            mask_glimpses = self.mask_glimpses

        if mask_logits is None:
            mask_logits = self.mask_logits

        hy, cy = self.lstm(x, h_in)
        g_l, h_out = hy, (hy, cy)

        for i in range(self.n_glimpses):
            ref, logits = self.glimpse(g_l, context)
            if mask_glimpses:
                logits = ops.tensorops.where(logit_mask, ops.Fill(-np.inf), logits)
            g_l = ops.Reshape()(ops.BatchMatMul()(ref, ops.ExpandDims()(self.sm(logits), 2)), (-1, self.hidden_dim))
        _, logits = self.pointer(g_l, context)

        if mask_logits:
            logits = ops.tensorops.where(logit_mask, ops.Fill(-np.inf), logits)

        return logits, h_out

    def construct(self, decoder_input, embedded_inputs, hidden, context, eval_tours=None):
        """
        Args:
            decoder_input: The initial input to the decoder
                size is [batch_size x embedding_dim]. Trainable parameter.
            embedded_inputs: [sourceL x batch_size x embedding_dim]
            hidden: the prev hidden state, size is [batch_size x hidden_dim]. 
                Initially this is set to (enc_h[-1], enc_c[-1])
            context: encoder outputs, [sourceL x batch_size x hidden_dim] 
        """

        batch_size = context.shape[1]
        outputs = []
        selections = []
        steps = range(embedded_inputs.shape[0])
        idxs = None
        mask = ops.Fill()(ops.DType()(embedded_inputs), embedded_inputs.shape[1:], 0)

        for i in steps:
            hidden, log_p, probs, mask = self.recurrence(decoder_input, hidden, mask, idxs, i, context)
            idxs = self.decode(
                probs,
                mask
            ) if eval_tours is None else eval_tours[:, i]

            idxs = idxs.detach()

            decoder_input = ops.Gather()(embedded_inputs,
                                          ops.ExpandDims()(ops.ExpandDims()(idxs, 0), -1)).squeeze()

            outputs.append(log_p)
            selections.append(idxs)
        return (ops.Stack(axis=1)(outputs), ops.Stack(axis=1)(selections)), hidden

    def decode(self, probs, mask):
        if self.decode_type == "greedy":
            _, idxs = ops.Max(axis=1)(probs)
            assert not ops.tensorops.Any()(ops.Gather()(mask, ops.ExpandDims()(idxs, -1))), \
                "Decode greedy: infeasible action has maximum probability"
        elif self.decode_type == "sampling":
            idxs = ops.tensorops.Multinomial()(ops.Exp(probs), 1).squeeze()
            while ops.tensorops.Any()(ops.Gather()(mask, ops.ExpandDims()(idxs, -1))):
                print(' [!] resampling due to race condition')
                idxs = ops.tensorops.Multinomial()(ops.Exp(probs), 1).squeeze()
        else:
            assert False, "Unknown decode type"

        return idxs

class CriticNetworkLSTM(nn.Cell):
    """Useful as a baseline in REINFORCE updates"""
    def __init__(self,
            embedding_dim,
            hidden_dim,
            n_process_block_iters,
            tanh_exploration,
            use_tanh):
        super(CriticNetworkLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_process_block_iters = n_process_block_iters

        self.encoder = Encoder(embedding_dim, hidden_dim)
        
        self.process_block = Attention(hidden_dim, use_tanh=use_tanh, C=tanh_exploration)
        self.sm = nn.Softmax(dim=1)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs):
        """
        Args:
            inputs: [embedding_dim x batch_size x sourceL] of embedded inputs
        """
        inputs = inputs.transpose(0, 1).contiguous()

        encoder_hx = self.encoder.init_hx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)
        encoder_cx = self.encoder.init_cx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)
        
        # encoder forward pass
        enc_outputs, (enc_h_t, enc_c_t) = self.encoder(inputs, (encoder_hx, encoder_cx))
        
        # grab the hidden state and process it via the process block 
        process_block_state = enc_h_t[-1]
        for i in range(self.n_process_block_iters):
            ref, logits = self.process_block(process_block_state, enc_outputs)
            process_block_state = ops.BatchMatMul(ref, self.sm(logits).unsqueeze(2)).squeeze(2)
        # produce the final scalar output
        out = self.decoder(process_block_state)
        return out

class PointerNetwork(nn.Cell):
    def __init__(self, embedding_dim, hidden_dim, problem, n_encode_layers=None,
                 tanh_clipping=10., mask_inner=True, mask_logits=True, normalization=None, **kwargs):
        super(PointerNetwork, self).__init__()

        self.problem = problem
        assert problem.NAME == "tsp", "Pointer Network only supported for TSP"
        self.input_dim = 2

        self.encoder = Encoder(embedding_dim, hidden_dim)
        self.decoder = Decoder(embedding_dim, hidden_dim, tanh_exploration=tanh_clipping,
                               use_tanh=tanh_clipping > 0, n_glimpses=1, mask_glimpses=mask_inner,
                               mask_logits=mask_logits)

        # Trainable initial hidden states
        std = 1. / math.sqrt(embedding_dim)
        self.decoder_in_0 = nn.Parameter(initializer.Uniform(-std, std)((embedding_dim,)))
        self.embedding = nn.Parameter(initializer.Uniform(-std, std)((self.input_dim, embedding_dim)))

    def set_decode_type(self, decode_type):
        self.decoder.decode_type = decode_type

    def construct(self, inputs, eval_tours=None, return_pi=False):
        batch_size, graph_size, input_dim = inputs.shape

        embedded_inputs = ops.MatMul()(
            inputs.transpose(0, 1).reshape(-1, input_dim),
            self.embedding
        ).reshape(graph_size, batch_size, -1)

        # query the actor net for the input indices 
        # making up the output, and the pointer attn 
        _log_p, pi = self._inner(embedded_inputs, eval_tours)

        cost, mask = self.problem.get_costs(inputs, pi)
        # Log likelihood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll = self._calc_log_likelihood(_log_p, pi, mask)
        if return_pi:
            return cost, ll, pi

        return cost, ll

    def _calc_log_likelihood(self, _log_p, a, mask):

        # Get log_p corresponding to selected actions
        log_p = ops.Gather(2)(_log_p, ops.Reshape()(a, (-1, 1))).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p = ops.ScatterNd()(log_p, mask, ops.Fill()(ops.Shape()(log_p), 0.0))

        assert ops.ReduceAll()(log_p > -1000), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return ops.ReduceSum()(log_p, 1)

    def _inner(self, inputs, eval_tours=None):

        encoder_hx = encoder_cx = ops.Zeros()(inputs.shape[1], self.encoder.hidden_dim)

        # encoder forward pass
        enc_h, (enc_h_t, enc_c_t) = self.encoder(inputs, (encoder_hx, encoder_cx))

        dec_init_state = (enc_h_t[-1], enc_c_t[-1])

        # repeat decoder_in_0 across batch
        decoder_input = ops.Tile()(self.decoder_in_0.unsqueeze(0), (inputs.shape[1], 1))

        (pointer_probs, input_idxs), dec_hidden_t = self.decoder(decoder_input,
                                                                 inputs,
                                                                 dec_init_state,
                                                                 enc_h,
                                                                 eval_tours)

        return pointer_probs, input_idxs