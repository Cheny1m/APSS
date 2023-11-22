import math
import numpy as np

import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms
from mindspore.common.initializer import initializer, Uniform

class SkipConnection(nn.Cell):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def construct(self, input):
        return input + self.module(input)

class MultiHeadAttention(nn.Cell):
    def __init__(self, n_heads, input_dim, embed_dim, val_dim=None, key_dim=None):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)

        self.W_query = ms.Parameter(initializer(Uniform(1. / math.sqrt(key_dim)), [n_heads, input_dim, key_dim], ms.float32)) 
        self.W_key = ms.Parameter(initializer(Uniform(1. / math.sqrt(key_dim)), [n_heads, input_dim, key_dim], ms.float32)) 
        self.W_val = ms.Parameter(initializer(Uniform(1. / math.sqrt(val_dim)), [n_heads, input_dim, val_dim], ms.float32))
        self.W_out = ms.Parameter(initializer(Uniform(1. / math.sqrt(embed_dim)), [n_heads, val_dim, embed_dim], ms.float32))

    def construct(self, q, h=None, mask=None):
        if h is None:
            h = q  # compute self-attention

        batch_size, graph_size, input_dim = h.shape
        n_query = q.shape[1]
        assert q.shape[0] == batch_size
        assert q.shape[2] == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        Q = ops.matmul(qflat, self.W_query).view(shp_q)
        K = ops.matmul(hflat, self.W_key).view(shp)
        V = ops.matmul(hflat, self.W_val).view(shp)

        compatibility = self.norm_factor * ops.matmul(Q, K.swapaxes(2, 3))

        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = ops.softmax(compatibility, axis=-1)

        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc
        
        heads = ops.matmul(attn, V)
        out = ops.reshape(ops.permute(heads,(1, 2, 0, 3)), (-1, self.n_heads * self.val_dim))
        W_out = self.W_out.view(-1, self.embed_dim)
        out = ops.mm(out, W_out).view(batch_size, n_query, self.embed_dim)

        return out

class Normalization(nn.Cell):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)
        self.normalizer = normalizer_class(embed_dim, affine=True)

    def construct(self, input):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.shape[-1])).view(*input.shape)
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input
        
    
class MultiHeadAttentionLayer(nn.SequentialCell):
    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.SequentialCell(
                    nn.Dense(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Dense(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Dense(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )

class GraphAttentionEncoder(nn.Cell):
    def __init__(
        self,
        n_heads,
        embed_dim,
        n_layers,
        node_dim=None,
        normalization='batch',
        feed_forward_hidden=512
    ):
        
        super(GraphAttentionEncoder, self).__init__()

        self.init_embed = nn.Dense(node_dim, embed_dim) if node_dim is not None else None

        # layers = []
        # for _ in range(n_layers):
        #     layers.append(MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization))
        # self.layers = nn.SequentialCell(layers)
        self.layers = nn.SequentialCell(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))

    def construct(self, x, mask=None):
        assert mask is None, "TODO mask not yet supported!"
        h = self.init_embed(x.view(-1, x.shape[-1])).view(*x.shape[:2], -1) if self.init_embed is not None else x
        h = self.layers(h)

        return h, h.mean(axis = 1)