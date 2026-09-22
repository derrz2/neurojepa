from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def resolve_mim_input(batch) -> torch.Tensor:
    x = batch
    if isinstance(x, tuple):
        x = x[0]
    if isinstance(x, list):
        x = x[0]
    if not torch.is_tensor(x):
        raise ValueError(f"Unsupported masked-model input type: {type(x)}")
    if x.ndim != 3:
        raise ValueError(f"Expected masked-model input shape (B, C, T), got {tuple(x.shape)}")
    return x


def pad_to_patch_size(x: torch.Tensor, patch_size: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[int, int]]:
    patch_c, patch_t = int(patch_size[0]), int(patch_size[1])
    pad_c = (-x.shape[-2]) % patch_c
    pad_t = (-x.shape[-1]) % patch_t
    if pad_c == 0 and pad_t == 0:
        return x, (x.shape[-2] // patch_c, x.shape[-1] // patch_t)
    padded = F.pad(x.unsqueeze(1), (0, pad_t, 0, pad_c)).squeeze(1)
    return padded, (padded.shape[-2] // patch_c, padded.shape[-1] // patch_t)


def patchify_channel_time(x: torch.Tensor, patch_size: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[int, int]]:
    x, grid_size = pad_to_patch_size(x, patch_size)
    b, c, t = x.shape
    patch_c, patch_t = int(patch_size[0]), int(patch_size[1])
    x = x.reshape(b, grid_size[0], patch_c, grid_size[1], patch_t)
    x = x.permute(0, 1, 3, 2, 4).reshape(b, grid_size[0] * grid_size[1], patch_c * patch_t)
    return x, grid_size


def random_masking(x: torch.Tensor, mask_ratio: float):
    n, l, d = x.shape
    len_keep = max(1, int(round(l * (1.0 - float(mask_ratio)))))

    noise = torch.rand(n, l, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, d))

    mask = torch.ones(n, l, device=x.device, dtype=x.dtype)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore, ids_keep


def random_binary_mask(batch_size: int, length: int, mask_ratio: float, device) -> torch.Tensor:
    noise = torch.rand(batch_size, length, device=device)
    num_mask = max(1, int(round(length * float(mask_ratio))))
    ids = torch.argsort(noise, dim=1)
    mask = torch.zeros(batch_size, length, device=device, dtype=torch.bool)
    mask.scatter_(1, ids[:, :num_mask], True)
    return mask
