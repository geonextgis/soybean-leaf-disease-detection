import math
from typing import List, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..utils.utils import cat_keep_shapes, uncat_with_shapes


# RoPE-related functions:
def rope_rotate_half(x: Tensor) -> Tensor:
    # x: [x0  x1  x2  x3  x4  x5]
    # out: [-x5  -x4  -x3  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    return (x * cos) + (rope_rotate_half(x) * sin)


class LinearKMaskedBias(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        o = self.out_features
        assert o % 3 == 0
        if self.bias is not None:
            self.register_buffer("bias_mask", torch.full_like(self.bias, fill_value=math.nan))
            
    def forward(self, input: Tensor) -> Tensor:
        masked_bias = self.bias * self.bias_mask.to(self.bias.dtype) if self.bias is not None else None
        return F.linear(input, self.weight, masked_bias)
    

# def SelfAttention(nn.Module):
#     def __init__(
#         self,
#         dim: int,
#         num_heads: int = 8,
#         qkv_bias: bool = False,
#         proj_bias: bool = True,
#         attn_drop: float = 0.0,
#         proj_drop: float = 0.0,
#         mask_k_bias: bool = False,
#         device = None,
#     ) -> None:
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim**-0.5
        
#         linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
#         self.qkv = linear_class(dim, dim * 3, bias=qkv_bias, device=device)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
#         self.proj_drop = nn.Dropout(proj_drop)
        
#     def apply_rope(self, q: Tenosr, k: Tensor, rope: Tensor):
#         q_dtype = q.dtype
#         k_dtype = k.dtype
#         sin, cos = rope
#         rope_dtype = sin.dtype
#         q = q.to(dtype=rope_dtype)
#         k = k.to(dtype=rope_dtype)
#         N = q.shape[-2]
        