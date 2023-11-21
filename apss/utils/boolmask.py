import mindspore.ops as ops
import mindspore as ms
from mindspore import Tensor
from mindspore import numpy as mnp

def _pad_mask(mask):
    # By taking -size % 8, we get 0 if exactly divisible by 8
    # and required padding otherwise (i.e. -1 % 8 = 7 pad)
    pad = -mask.size(-1) % 8
    if pad != 0:
        mask = ops.pad(mask, [0, pad])
    return mask, mask.size(-1) // 8

# before
# def _mask_bool2byte(mask):
#     assert mask.dtype == ms.uint8
#     # assert (mask <= 1).all()  # Precondition, disabled for efficiency
#     mask, d = _pad_mask(mask)
#     return (mask.view(*mask.size()[:-1], d, 8) << torch.arange(8, out=mask.new())).sum(-1, dtype=torch.uint8)

# after
def _mask_bool2byte(mask):
    assert mask.dtype == ms.uint8
    # assert (mask <= 1).all()

    mask, d = _pad_mask(mask)
    
    arange = ms.ops.range(0, 8, dtype=ms.int64)
    mask_shifted = mask.view(*mask.size()[:-1], d, 8) << arange
    return mask_shifted.sum(-1, dtype=ms.uint8)

# before
# def _mask_byte2long(mask):
#     assert mask.dtype == torch.uint8
#     mask, d = _pad_mask(mask)
#     # Note this corresponds to a temporary factor 8
#     # memory overhead by converting to long before summing
#     # Alternatively, aggregate using for loop
#     return (mask.view(*mask.size()[:-1], d, 8).long() << (torch.arange(8, dtype=torch.int64, device=mask.device) * 8)).sum(-1)
def _mask_byte2long(mask):
    assert mask.dtype == ms.uint8
    mask, d = _pad_mask(mask)

    # Note this corresponds to a temporary factor 8
    # memory overhead by converting to long before summing
    # Alternatively, aggregate using for loop
    
    arange = ms.ops.range(0, 8, dtype=ms.int64) * 8
    mask_shifted = mask.view(*mask.size()[:-1], d, 8).astype(ms.int64) << arange
    return mask_shifted.sum(-1)




def mask_bool2long(mask):
    assert mask.dtype == ms.uint8
    return _mask_byte2long(_mask_bool2byte(mask))

# before
# def _mask_long2byte(mask, n=None):
#     if n is None:
#         n = 8 * mask.size(-1)
#     return (mask[..., None] >> (torch.arange(8, out=mask.new()) * 8))[..., :n].to(torch.uint8).view(*mask.size()[:-1], -1)[..., :n]

# after
# def _mask_long2byte(mask, n=None):
#     if n is None:
#         n = 8 * mask.shape[-1]
        
#     arange = Tensor([i for i in range(8)], mask.dtype) * 8
#     mask_shifted = (mask[..., None] >> arange)[..., :n].astype(ms.uint8).reshape(*mask.size()[:-1], -1)[..., :n]
#     return mask_shifted
def _mask_long2byte(mask, n=None):
    if n is None:
        n = 8 * mask.shape[-1]
        
    arange = Tensor([2 ** (i * 8) for i in range(8)], mask.dtype)
    
    # Instead of right-shifting, divide the mask by each value in arange.
    mask_divided = ops.Div()(mask[..., None], arange)
    
    # Convert the result of the division to integers
    mask_floor_divided = mask_divided[..., :n].astype(ms.uint8).reshape(*mask.shape[:-1], -1)[..., :n]
    
    return mask_floor_divided
    
def _mask_byte2bool(mask, n=None):
    if n is None:
        n = 8 * mask.shape[-1]
        
    # Replace the left-shift by multiplying with 2 raised to the power of arange
    arange = Tensor([2 ** i for i in range(8)], dtype=mask.dtype)
    
    # Simulate the left-shift by multiplying with appropriate powers of 2
    mask_filtered = ops.Mod()(mask[..., None], (arange * 2)) // arange
    
    # Reshape and filter the mask
    mask_reshaped = mask_filtered.reshape(*mask.shape[:-1], -1)[..., :n]
    
    return mask_reshaped > 0


def mask_long2bool(mask, n=None):
    assert mask.dtype == ms.int64
    # print("mask:",mask.shape,"n:",n)
    return _mask_byte2bool(_mask_long2byte(mask), n=n)

def mask_long_scatter(mask, values, check_unset=True):
    
    assert mask.shape[:-1] == values.shape
    rng = mnp.arange(mask.shape[-1], dtype=mask.dtype)
    values_ = values[..., None]  

    where = (values_ >= (rng * 64)) & (values_ < ((rng + 1) * 64))
    
    power_shift = mnp.power(2,(values_ % 64))
    if check_unset:
        assert not ((mask & (where * power_shift)) > 0).any()

    return mask | (where * power_shift)