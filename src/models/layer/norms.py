from functools import partial

import torch.nn as nn


def build_norm_layer(norm_type="layernorm", eps=1e-6):
    norm_type = str(norm_type or "layernorm").lower()

    if norm_type in ("layernorm", "ln"):
        return partial(nn.LayerNorm, eps=eps)
    if norm_type in ("rmsnorm", "rms"):
        return partial(nn.RMSNorm, eps=eps)
    raise ValueError(f"Unsupported norm type: {norm_type}")
