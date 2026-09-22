import math
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn


class RopePositionEmbedding(nn.Module):
    """
    2D axial RoPE generator for channel-time token coordinates.

    Returns sin/cos tensors with shape (..., rotary_dim), where rotary_dim is a
    multiple of 4 (2 dims per axis per frequency).
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        rotary_dim_ratio: float = 1.0,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        self.head_dim = embed_dim // num_heads
        rotary_dim = int(self.head_dim * rotary_dim_ratio)
        rotary_dim = max(0, min(rotary_dim, self.head_dim))
        # 2 axes + pairwise rotation => multiples of 4.
        self.rotary_dim = (rotary_dim // 4) * 4
        self.n_freq = self.rotary_dim // 4

        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.dtype = dtype

        periods_shape = max(1, self.n_freq)
        self.register_buffer(
            "periods",
            torch.empty(periods_shape, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(
        self,
        coords: Tensor,
        *,
        spatial_shape: tuple[int, int] | None = None,
    ) -> tuple[Tensor, Tensor]:
        if self.rotary_dim == 0:
            empty = coords.new_zeros(*coords.shape[:-1], 0)
            return empty, empty
        if coords.shape[-1] != 2:
            raise ValueError(f"coords must have last dim=2, got {coords.shape}")

        coords = self._normalize_coords(coords, spatial_shape=spatial_shape)
        coords = self._augment_coords(coords)

        periods = self.periods[: self.n_freq].to(device=coords.device, dtype=coords.dtype)
        angles = 2.0 * math.pi * coords[..., :, None] / periods[None, None, :]
        # (..., 2, n_freq) -> (..., 2, 2*n_freq) by repeating each frequency twice.
        angles = torch.repeat_interleave(angles, repeats=2, dim=-1)
        # (..., 2, 2*n_freq) -> (..., 4*n_freq)
        angles = angles.reshape(*coords.shape[:-1], self.rotary_dim)

        sin = torch.sin(angles)
        cos = torch.cos(angles)
        return sin, cos

    def _normalize_coords(
        self, coords: Tensor, *, spatial_shape: tuple[int, int] | None
    ) -> Tensor:
        if spatial_shape is not None:
            c_len, t_len = spatial_shape
            dims = torch.tensor([max(c_len, 1), max(t_len, 1)], device=coords.device, dtype=coords.dtype)
        else:
            max_abs = coords.abs().amax(dim=tuple(range(coords.ndim - 1)), keepdim=False)
            dims = torch.clamp(max_abs, min=1.0)

        if self.normalize_coords == "separate":
            denom = dims
        elif self.normalize_coords == "max":
            denom = torch.full_like(dims, dims.max())
        elif self.normalize_coords == "min":
            denom = torch.full_like(dims, dims.min())
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")

        coords = (coords + 0.5) / denom
        coords = 2.0 * coords - 1.0
        return coords

    def _augment_coords(self, coords: Tensor) -> Tensor:
        if not self.training:
            return coords

        dd = {"device": coords.device, "dtype": coords.dtype}
        if self.shift_coords is not None:
            shift = torch.empty(2, **dd).uniform_(-self.shift_coords, self.shift_coords)
            coords = coords + shift

        if self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_min = -jitter_max
            jitter = torch.empty(2, **dd).uniform_(jitter_min, jitter_max).exp()
            coords = coords * jitter

        if self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale = torch.empty(1, **dd).uniform_(rescale_min, rescale_max).exp()
            coords = coords * rescale
        return coords

    def _init_weights(self) -> None:
        device = self.periods.device
        dtype = self.dtype if self.dtype is not None else torch.float32
        n = max(1, self.n_freq)

        if self.base is not None:
            denom = max(self.rotary_dim, 1)
            periods = self.base ** (2 * torch.arange(n, device=device, dtype=dtype) / denom)
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, n, device=device, dtype=dtype)
            periods = base**exponents
            periods = periods / base
            periods = periods * self.max_period

        self.periods.data = periods
