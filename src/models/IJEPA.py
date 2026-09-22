import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .ijepa_utils import gather_patch_tokens, sample_target_masks
from .layer.vit_components import Block
from .ssl_utils import (
    build_probe,
    compute_collapse_metrics,
    compute_probe_loss,
    cosine_alignment,
    get_method_cfg,
    get_probe_flag,
)
from ..utils.optim import get_params_groups_with_decay, fuse_params_groups


class IJEPAPredictor(nn.Module):
    def __init__(
        self,
        *,
        num_patches,
        embed_dim,
        predictor_embed_dim,
        predictor_depth,
        predictor_num_heads,
        mlp_ratio,
        qkv_bias,
        qk_norm,
        drop_path_rate,
    ):
        super().__init__()
        self.input_proj = nn.Linear(embed_dim, predictor_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, int(num_patches), predictor_embed_dim))

        dpr = torch.linspace(0, float(drop_path_rate), int(predictor_depth)).tolist()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=predictor_embed_dim,
                    num_heads=int(predictor_num_heads),
                    mlp_ratio=float(mlp_ratio),
                    qkv_bias=bool(qkv_bias),
                    qk_norm=bool(qk_norm),
                    drop_path=float(dpr[i]),
                    use_rope=False,
                )
                for i in range(int(predictor_depth))
            ]
        )
        self.norm = nn.LayerNorm(predictor_embed_dim)
        self.pred = nn.Linear(predictor_embed_dim, embed_dim)

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @staticmethod
    def _expand_indices(indices, batch_size, device):
        if indices.ndim == 1:
            indices = indices.unsqueeze(0).expand(batch_size, -1)
        elif indices.ndim != 2:
            raise ValueError(f"Expected indices to have shape (M,) or (B, M), got {tuple(indices.shape)}")
        return indices.to(device=device, dtype=torch.long)

    def forward(self, context_tokens, context_indices, target_indices):
        batch_size = context_tokens.shape[0]
        context_indices = self._expand_indices(context_indices, batch_size, context_tokens.device)
        target_indices = self._expand_indices(target_indices, batch_size, context_tokens.device)
        target_len = target_indices.shape[1]
        pos_embed = self.pos_embed.expand(batch_size, -1, -1)

        context_pos = gather_patch_tokens(pos_embed, context_indices)
        target_pos = gather_patch_tokens(pos_embed, target_indices)

        context_tokens = self.input_proj(context_tokens) + context_pos
        target_tokens = self.mask_token.expand(batch_size, target_len, -1) + target_pos

        x = torch.cat([context_tokens, target_tokens], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.pred(x[:, -target_len:])


class IJEPAModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.cfg = args
        self.ijepa_cfg = get_method_cfg(args, "ijepa")

        self.backbone = build_backbone(args, downstream=False)
        self.target_encoder = build_backbone(args, downstream=False)
        self.target_encoder.load_state_dict(self.backbone.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        self.target_encoder.eval()

        if not hasattr(self.backbone, "forward_tokens"):
            raise ValueError("I-JEPA requires a backbone with token-level forward support.")

        self.loss_type = str(getattr(self.ijepa_cfg, "loss_type", "smooth_l1")).lower()
        self.normalize_targets = bool(getattr(self.ijepa_cfg, "normalize_targets", True))
        self.num_targets = int(getattr(self.ijepa_cfg, "num_targets", 4))
        self.scale_range = tuple(getattr(self.ijepa_cfg, "target_scale_range", (0.10, 0.20)))
        self.aspect_ratio_range = tuple(getattr(self.ijepa_cfg, "target_aspect_ratio_range", (0.5, 2.0)))
        self.min_context_patches = int(getattr(self.ijepa_cfg, "min_context_patches", 4))

        embed_dim = int(self.backbone.embed_dim)
        self.predictor = IJEPAPredictor(
            num_patches=int(self.backbone.num_patches),
            embed_dim=embed_dim,
            predictor_embed_dim=int(getattr(self.ijepa_cfg, "predictor_embed_dim", 384)),
            predictor_depth=int(getattr(self.ijepa_cfg, "predictor_depth", 4)),
            predictor_num_heads=int(getattr(self.ijepa_cfg, "predictor_num_heads", 6)),
            mlp_ratio=float(getattr(self.ijepa_cfg, "predictor_mlp_ratio", 4.0)),
            qkv_bias=bool(getattr(args.model, "qkv_bias", True)),
            qk_norm=bool(getattr(args.model, "qk_norm", True)),
            drop_path_rate=float(getattr(self.ijepa_cfg, "predictor_drop_path_rate", 0.0)),
        )

        if get_probe_flag(args):
            self.probe = build_probe(embed_dim)

    def _normalize_token_targets(self, x):
        if not self.normalize_targets:
            return x.float()
        return F.layer_norm(x.float(), (x.shape[-1],))

    def _compute_ssl_loss(self, pred_tokens, target_tokens):
        pred_tokens = self._normalize_token_targets(pred_tokens)
        target_tokens = self._normalize_token_targets(target_tokens)
        if self.loss_type == "mse":
            return F.mse_loss(pred_tokens, target_tokens)
        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred_tokens, target_tokens)
        raise ValueError(f"Unsupported I-JEPA loss_type: {self.loss_type}")

    def forward(self, input_batch):
        x, y = input_batch if get_probe_flag(self.cfg) else (input_batch, None)
        if isinstance(x, (list, tuple)):
            if len(x) != 1:
                raise ValueError(f"I-JEPA expects a single view, got {len(x)} views.")
            x = x[0]

        with torch.no_grad():
            teacher_tokens, grid_size = self.target_encoder.forward_tokens(x, return_grid=True)
            context_mask, target_indices = sample_target_masks(
                grid_size,
                num_targets=self.num_targets,
                scale_range=self.scale_range,
                aspect_ratio_range=self.aspect_ratio_range,
                min_context_patches=self.min_context_patches,
                device=x.device,
            )
            target_tokens = gather_patch_tokens(teacher_tokens, target_indices)

        context_tokens, context_indices = self.backbone.forward_tokens(
            x,
            keep_mask=context_mask,
            return_keep_indices=True,
        )
        pred_tokens = self.predictor(context_tokens, context_indices, target_indices)
        ijepa_loss = self._compute_ssl_loss(pred_tokens, target_tokens)

        online_embed = self.backbone(x)
        prob_loss = ijepa_loss.new_zeros(())
        if get_probe_flag(self.cfg):
            prob_loss = compute_probe_loss(self.probe, online_embed, y, 1)

        loss = ijepa_loss + prob_loss
        alignment = cosine_alignment(pred_tokens.mean(dim=1), target_tokens.mean(dim=1))
        collapse_metrics = compute_collapse_metrics(online_embed.float())

        return loss, ijepa_loss, prob_loss, alignment, collapse_metrics

    @torch.no_grad()
    def update_teacher(self, momentum):
        student_params = list(self.backbone.parameters())
        teacher_params = list(self.target_encoder.parameters())
        torch._foreach_mul_(teacher_params, momentum)
        torch._foreach_add_(teacher_params, student_params, alpha=1 - momentum)

    def get_params_groups(self):
        params_groups = get_params_groups_with_decay(
            model=self,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        for group in fused_params_groups:
            group["foreach"] = True
        return fused_params_groups
