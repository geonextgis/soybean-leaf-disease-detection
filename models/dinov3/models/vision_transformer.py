from functools import partial
import logging
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.init

from ..layers.layer_scale import LayerScale
