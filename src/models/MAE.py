from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
from timm.layers import trunc_normal_

from .backbone import VisionTransformer, build_backbone
from .mim_utils import patchify_channel_time, random_masking, resolve_mim_input
from ..utils.optim import fuse_params_groups, get_params_groups_with_decay
from .layer.vit_components import Block


class MAE_VIT(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.cfg = args
        self.mae_cfg = args.model.mae

        self.backbone = build_backbone(args, downstream=False)
        if not isinstance(self.backbone, VisionTransformer):
            raise TypeError("MAE currently supports ViT backbone only.")
        if self.backbone.num_reg_tokens > 0:
            raise ValueError("MAE does not support reg_tokens > 0. Please set reg_tokens=0 for MAE.")

        self.mask_ratio = float(getattr(self.mae_cfg, "mask_ratio", 0.75))
        self.norm_pix_loss = bool(getattr(self.mae_cfg, "norm_pix_loss", True))

        self.decoder_embed_dim = int(getattr(self.mae_cfg, "decoder_embed_dim", 512))
        self.decoder_depth = int(getattr(self.mae_cfg, "decoder_depth", 8))
        self.decoder_num_heads = int(getattr(self.mae_cfg, "decoder_num_heads", 16))

        patch_dim = int(self.backbone.patch_size[0] * self.backbone.patch_size[1])

        self.decoder_embed = nn.Linear(self.backbone.embed_dim, self.decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(
                1,
                self.backbone.num_patches + self.backbone.num_prefix_tokens,
                self.decoder_embed_dim,
            )
        )
        decoder_norm = partial(nn.LayerNorm, eps=1e-6)
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    dim=self.decoder_embed_dim,
                    num_heads=self.decoder_num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    qk_norm=False,
                    proj_bias=True,
                    init_values=None,
                    proj_drop=0.0,
                    attn_drop=0.0,
                    drop_path=0.0,
                    norm_layer=decoder_norm,
                    qk_norm_layer=decoder_norm,
                    act_layer=nn.GELU,
                )
                for _ in range(self.decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(self.decoder_embed_dim)
        self.decoder_pred = nn.Linear(self.decoder_embed_dim, patch_dim, bias=True)

        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.decoder_pos_embed, std=0.02)

    def _resize_decoder_pos_embed(self, grid_size):
        prefix_n = self.backbone.num_prefix_tokens
        if tuple(grid_size) == tuple(self.backbone.base_grid_size):
            return self.decoder_pos_embed

        prefix_pos = self.decoder_pos_embed[:, :prefix_n] if prefix_n > 0 else None
        patch_pos = self.decoder_pos_embed[:, prefix_n:]
        patch_pos = patch_pos.reshape(
            1,
            self.backbone.base_grid_size[0],
            self.backbone.base_grid_size[1],
            self.decoder_embed_dim,
        ).permute(0, 3, 1, 2)
        patch_pos = nn.functional.interpolate(
            patch_pos,
            size=grid_size,
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], self.decoder_embed_dim)
        if prefix_pos is None:
            return patch_pos
        return torch.cat([prefix_pos, patch_pos], dim=1)

    def forward_encoder(self, x: torch.Tensor):
        target_dtype = self.backbone.pos_embed.dtype if self.backbone.pos_embed is not None else self.backbone.patch_embed.proj.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(dtype=target_dtype)

        x, grid_size = self.backbone.patch_embed(x)
        pos_embed = self.backbone._resize_pos_embed(grid_size)
        prefix_n = self.backbone.num_prefix_tokens
        patch_pos = pos_embed[:, prefix_n:] if pos_embed is not None and prefix_n > 0 else pos_embed
        if patch_pos is not None:
            x = x + patch_pos

        x, mask, ids_restore, _ = random_masking(x, self.mask_ratio)

        if prefix_n > 0:
            prefix_tokens = []
            if self.backbone.cls_token is not None:
                prefix_tokens.append(self.backbone.cls_token.expand(x.shape[0], -1, -1))
            if self.backbone.reg_token is not None:
                prefix_tokens.append(self.backbone.reg_token.expand(x.shape[0], -1, -1))
            prefix_tokens = torch.cat(prefix_tokens, dim=1)
            if pos_embed is not None:
                prefix_tokens = prefix_tokens + pos_embed[:, :prefix_n]
            x = torch.cat([prefix_tokens, x], dim=1)

        x = self.backbone.pos_drop(x)
        x = self.backbone.norm_pre(x)
        for block in self.backbone.blocks:
            x = block(x)
        x = self.backbone.norm(x)
        return x, mask, ids_restore, grid_size

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor, grid_size):
        x = self.decoder_embed(x)
        prefix_n = self.backbone.num_prefix_tokens

        visible_tokens = x[:, prefix_n:, :]
        num_mask = ids_restore.shape[1] - visible_tokens.shape[1]
        if num_mask > 0:
            mask_tokens = self.mask_token.repeat(x.shape[0], num_mask, 1)
            visible_tokens = torch.cat([visible_tokens, mask_tokens], dim=1)
        visible_tokens = torch.gather(
            visible_tokens,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, visible_tokens.shape[-1]),
        )

        if prefix_n > 0:
            x = torch.cat([x[:, :prefix_n, :], visible_tokens], dim=1)
        else:
            x = visible_tokens

        x = x + self._resize_decoder_pos_embed(grid_size)
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x[:, prefix_n:, :]

    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        target, _ = patchify_channel_time(imgs, self.backbone.patch_size)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6).sqrt()

        loss = (pred - target).pow(2).mean(dim=-1)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)

    def forward(self, inputs):
        x = resolve_mim_input(inputs)
        latent, mask, ids_restore, grid_size = self.forward_encoder(x)
        pred = self.forward_decoder(latent, ids_restore, grid_size)
        loss = self.forward_loss(x, pred, mask)
        return loss, loss.detach(), mask.mean()

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
