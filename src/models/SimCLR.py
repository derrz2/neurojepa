import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .ssl_utils import (
    build_mlp_head,
    build_probe,
    compute_collapse_metrics,
    compute_probe_loss,
    cosine_alignment,
    gather_with_grad,
    get_method_cfg,
    get_probe_flag,
    resolve_ssl_input,
)
from ..utils.optim import get_params_groups_with_decay, fuse_params_groups


class SimCLRModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.cfg = args
        self.simclr_cfg = get_method_cfg(args, "simclr")

        self.backbone = build_backbone(args, downstream=False)
        embed_dim = int(self.backbone.embed_dim)

        proj_hidden_dim = int(getattr(self.simclr_cfg, "proj_hidden_dim", 2048))
        proj_output_dim = int(getattr(self.simclr_cfg, "proj_output_dim", 128))
        proj_layers = int(getattr(self.simclr_cfg, "proj_num_layers", 2))
        self.temperature = float(getattr(self.simclr_cfg, "temperature", 0.2))
        if self.temperature <= 0.0:
            raise ValueError(f"SimCLR temperature must be > 0, got {self.temperature}")

        self.projector = build_mlp_head(embed_dim, proj_hidden_dim, proj_output_dim, proj_layers)

        if get_probe_flag(args):
            self.probe = build_probe(embed_dim)

    def forward(self, input_batch):
        views, y = resolve_ssl_input(input_batch, probe_val=get_probe_flag(self.cfg), min_views=2)
        x1, x2 = views[0], views[1]
        bs = x1.shape[0]

        feats = self.backbone(torch.cat([x1, x2], dim=0))
        feats = feats.to(next(self.projector.parameters()).dtype)
        proj = self.projector(feats)
        z1 = F.normalize(proj[:bs].float(), dim=-1)
        z2 = F.normalize(proj[bs:].float(), dim=-1)

        simclr_loss = self._simclr_loss(z1, z2)

        prob_loss = simclr_loss.new_zeros(())
        if get_probe_flag(self.cfg):
            prob_loss = compute_probe_loss(self.probe, feats, y, 2)

        loss = simclr_loss + prob_loss
        alignment = cosine_alignment(z1, z2)
        collapse_metrics = compute_collapse_metrics(torch.cat([z1, z2], dim=0))

        return loss, simclr_loss, prob_loss, alignment, collapse_metrics

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

    def _simclr_loss(self, z1, z2):
        local = torch.cat([z1, z2], dim=0)
        gathered = gather_with_grad(local)

        local_n = local.shape[0]
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        offset = rank * local_n

        logits = (local @ gathered.T) / self.temperature
        logits = logits.float()

        self_idx = offset + torch.arange(local_n, device=logits.device)
        logits[torch.arange(local_n, device=logits.device), self_idx] = torch.finfo(logits.dtype).min

        half = local_n // 2
        targets = torch.cat(
            [
                offset + half + torch.arange(half, device=logits.device),
                offset + torch.arange(half, device=logits.device),
            ],
            dim=0,
        )
        return F.cross_entropy(logits, targets)
