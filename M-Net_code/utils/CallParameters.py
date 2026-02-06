import torch
import numpy as np
# `pip install thop`
from thop import profile
from thop import clever_format




def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('#'*20, '\n[Statistics Information]\nFLOPs: {}\nParams: {}\n'.format(flops, params), '#'*20)