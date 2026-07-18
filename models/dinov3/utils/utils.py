import logging
import os
import random
import subprocess
from typing import Callable, List, Optional, Tuple

from einops import rearrange
import numpy as np
import torch
from torch import Tensor, nn


def cat_keep_shapes(x_list: List[Tensor]) -> Tuple[Tensor, List[Tuple[int]], List[int]]:
    shapes = [x.shape for x in x_list]
    num_tokens = [x.select(dim=-1, index=0).numel() for x in x_list]
    flattened = [rearrange(x, "... d -> (...) d") for x in x_list]
    return torch.cat(flattened, dim=0), shapes, num_tokens

def uncat_with_shapes(flattened: Tensor, shapes: List[Tuple[int]], num_tokens: List[int]) -> List[Tensor]:
    outputs_splitted = torch.split_with_sizes(flattened, num_tokens, dim=0)
    shapes_adjusted = [shape[:-1] + torch.Size([flattened.shape[-1]]) for shape in shapes]
    outputs_reshaped = [o.reshape(shape) for o, shape in zip(outputs_splitted, shapes_adjusted)]
    return outputs_reshaped