import math
from typing import Optional, Tuple

import torch


def gather_patch_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if indices.ndim == 1:
        indices = indices.unsqueeze(0).expand(tokens.shape[0], -1)
    elif indices.ndim != 2:
        raise ValueError(f"Expected indices to have shape (M,) or (B, M), got {tuple(indices.shape)}")

    indices = indices.to(device=tokens.device, dtype=torch.long)
    gather_index = indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
    return torch.gather(tokens, dim=1, index=gather_index)


def sample_target_masks(
    grid_size: Tuple[int, int],
    *,
    num_targets: int,
    scale_range: Tuple[float, float],
    aspect_ratio_range: Tuple[float, float],
    min_context_patches: int = 4,
    max_attempts: int = 32,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_h, grid_w = int(grid_size[0]), int(grid_size[1])
    total_patches = grid_h * grid_w
    if total_patches <= min_context_patches:
        raise ValueError(f"Grid {grid_size} is too small for I-JEPA masking.")

    device = device or torch.device("cpu")
    min_scale, max_scale = float(scale_range[0]), float(scale_range[1])
    min_ar, max_ar = float(aspect_ratio_range[0]), float(aspect_ratio_range[1])
    min_area = max(1, int(round(total_patches * min_scale)))
    max_area = max(min_area, int(round(total_patches * max_scale)))

    occupied = torch.zeros((grid_h, grid_w), device=device, dtype=torch.bool)

    for _ in range(int(num_targets)):
        block_mask = None
        for _ in range(max_attempts):
            area = int(torch.randint(min_area, max_area + 1, (1,), device=device).item())
            aspect = float(torch.empty(1, device=device).uniform_(min_ar, max_ar).item())

            block_h = max(1, min(grid_h, int(round(math.sqrt(area / max(aspect, 1e-6))))))
            block_w = max(1, min(grid_w, int(round(math.sqrt(area * aspect)))))

            if block_h > grid_h or block_w > grid_w:
                continue

            top = int(torch.randint(0, grid_h - block_h + 1, (1,), device=device).item())
            left = int(torch.randint(0, grid_w - block_w + 1, (1,), device=device).item())

            candidate = torch.zeros_like(occupied)
            candidate[top : top + block_h, left : left + block_w] = True
            if torch.any(candidate & occupied):
                continue

            remaining = total_patches - int((occupied | candidate).sum().item())
            if remaining < int(min_context_patches):
                continue

            block_mask = candidate
            break

        if block_mask is None:
            break
        occupied |= block_mask

    if not torch.any(occupied):
        raise RuntimeError("Failed to sample any I-JEPA target blocks.")

    target_mask = occupied.flatten()
    context_mask = ~target_mask
    if int(context_mask.sum().item()) < int(min_context_patches):
        raise RuntimeError("I-JEPA masking left too few context patches.")

    return context_mask, target_mask.nonzero(as_tuple=False).flatten()
