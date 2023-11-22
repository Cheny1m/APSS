import numpy as np
import mindspore as ms
import mindspore.ops as ops

def torch_lexsort(keys, dim=-1):

  if keys[0].device_type == "GPU":
    # 使用GPU版本实现
    return _torch_lexsort_cuda(keys, dim)
  
  else:
    # 使用CPU版本实现  
    k_np = [k.asnumpy() for k in keys]
    idx = ms.Tensor(np.lexsort(k_np, axis=dim)) 
    return idx


def _torch_lexsort_cuda(keys, dim=-1):
    MIN_NUMEL_STABLE_SORT = 2049 
  # 实现轴转置和reshape
    reordered_keys = [ops.transpose(k, dim, -1) for k in keys]
    flat_keys = [ops.reshape(k, (-1,)) for k in reordered_keys]

    d = keys[0].shape[dim]
    numel = flat_keys[0].shape[0]
    batch_size = numel // d

    batch_key = ops.range(batch_size)
    batch_key = ops.reshape(batch_key, (batch_size, 1))
    batch_key = ops.tile(batch_key, (1, d))
    batch_key = ops.reshape(batch_key, (-1))

    flat_keys.append(batch_key)

    if numel < MIN_NUMEL_STABLE_SORT:
        # 实现数据复制
        n_rep = (MIN_NUMEL_STABLE_SORT + numel - 1) // numel
        rep_key = ops.range(n_rep)
        rep_key = ops.reshape(rep_key, (n_rep, 1))  
        rep_key = ops.tile(rep_key, (1, numel))
        rep_key = ops.reshape(rep_key, (-1))
        flat_keys = [ops.tile(k, (n_rep,)) for k in flat_keys]
        flat_keys.append(rep_key)

    # 实现排序
    idx = None
    for k in flat_keys:
        _, idx = ops.sort(k, axis=-1) if idx is None else ops.sort(k[idx], axis=-1)

    # 只返回需要的索引部分
    if numel < MIN_NUMEL_STABLE_SORT:
        idx = idx[:numel]

    idx = idx.view(reordered_keys[0].shape).transpose(dim, -1) % d

    return idx