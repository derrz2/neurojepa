import math
import torch.nn as nn
from timm.layers import Mlp


def _to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)


def _make_divisible(x, divisor=8):
    return int(math.ceil(x / divisor) * divisor)


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = _make_divisible((2 * hidden_features) / 3)
        bias = _to_2tuple(bias)
        drop = _to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features * 2, bias=bias[0])
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(drop[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop[1])

    def forward(self, x):
        x_gate, x_value = self.fc1(x).chunk(2, dim=-1)
        x = self.act(x_gate) * x_value
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def build_ffn_layer(ffn_layer="mlp"):
    ffn_layer = str(ffn_layer or "mlp").lower()

    if ffn_layer == "mlp":
        return Mlp
    if ffn_layer == "swiglu":
        return SwiGLUFFN
    raise ValueError(f"Unsupported ffn layer: {ffn_layer}")
