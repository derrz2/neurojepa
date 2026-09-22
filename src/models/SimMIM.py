from __future__ import annotations

import torch
import torch.nn as nn
from timm.layers import trunc_normal_

from .backbone import VisionTransformer, build_backbone
from .mim_utils import patchify_channel_time, random_binary_mask, resolve_mim_input
from ..utils.optim import fuse_params_groups, get_params_groups_with_decay


class SimMIM_VIT(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.cfg = args
        self.simmim_cfg = args.model.simmim

        self.backbone = build_backbone(args, downstream=False)
        if not isinstance(self.backbone, VisionTransformer):
            raise TypeError("SimMIM currently supports ViT backbone only.")
        if self.backbone.num_reg_tokens > 0:
            raise ValueError("SimMIM does not support reg_tokens > 0. Please set reg_tokens=0 for SimMIM.")

        self.mask_ratio = float(getattr(self.simmim_cfg, "mask_ratio", 0.6))
        self.loss_type = str(getattr(self.simmim_cfg, "loss_type", "l1")).lower()
        self.norm_pix_loss = bool(getattr(self.simmim_cfg, "norm_pix_loss", False))

        patch_dim = int(self.backbone.patch_size[0] * self.backbone.patch_size[1])
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.backbone.embed_dim))
        self.decoder = nn.Linear(self.backbone.embed_dim, patch_dim, bias=True)
        trunc_normal_(self.mask_token, std=0.02)

    def forward_encoder(self, x: torch.Tensor):
        target_dtype = self.backbone.pos_embed.dtype if self.backbone.pos_embed is not None else self.backbone.patch_embed.proj.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(dtype=target_dtype)

        x, grid_size = self.backbone.patch_embed(x)
        mask = random_binary_mask(x.shape[0], x.shape[1], self.mask_ratio, x.device)
        x = torch.where(mask.unsqueeze(-1), self.mask_token.expand(x.shape[0], x.shape[1], -1), x)
        x = self.backbone._apply_pos_embed(x, grid_size)
        x = self.backbone.forward_features(x, grid_size)[-1]
        return x[:, self.backbone.num_prefix_tokens :, :], mask

    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        target, _ = patchify_channel_time(imgs, self.backbone.patch_size)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6).sqrt()

        if self.loss_type == "l2":
            loss = (pred - target).pow(2).mean(dim=-1)
        else:
            loss = (pred - target).abs().mean(dim=-1)
        return (loss * mask.float()).sum() / mask.float().sum().clamp_min(1.0)

    def forward(self, inputs):
        x = resolve_mim_input(inputs)
        feats, mask = self.forward_encoder(x)
        pred = self.decoder(feats)
        loss = self.forward_loss(x, pred, mask)
        return loss, loss.detach(), mask.float().mean()

    def get_params_groups(self):
        params_groups = get_params_groups_with_decay(
            model=self,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups
