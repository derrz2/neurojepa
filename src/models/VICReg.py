import torch
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
    off_diagonal,
    resolve_ssl_input,
)
from ..utils.optim import get_params_groups_with_decay, fuse_params_groups


class VICRegModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.cfg = args
        self.vicreg_cfg = get_method_cfg(args, "vicreg")

        self.backbone = build_backbone(args, downstream=False)
        embed_dim = int(self.backbone.embed_dim)

        proj_hidden_dim = int(getattr(self.vicreg_cfg, "proj_hidden_dim", 2048))
        proj_output_dim = int(getattr(self.vicreg_cfg, "proj_output_dim", 256))
        proj_layers = int(getattr(self.vicreg_cfg, "proj_num_layers", 3))

        self.projector = build_mlp_head(embed_dim, proj_hidden_dim, proj_output_dim, proj_layers)

        self.sim_coeff = float(getattr(self.vicreg_cfg, "sim_coeff", 25.0))
        self.var_coeff = float(getattr(self.vicreg_cfg, "var_coeff", 25.0))
        self.cov_coeff = float(getattr(self.vicreg_cfg, "cov_coeff", 1.0))
        self.std_target = float(getattr(self.vicreg_cfg, "std_target", 1.0))
        self.eps = float(getattr(self.vicreg_cfg, "eps", 1e-4))

        if get_probe_flag(args):
            self.probe = build_probe(embed_dim)

    def forward(self, input_batch):
        views, y = resolve_ssl_input(input_batch, probe_val=get_probe_flag(self.cfg), min_views=2)
        x1, x2 = views[0], views[1]
        bs = x1.shape[0]

        feats = self.backbone(torch.cat([x1, x2], dim=0))
        feats = feats.to(next(self.projector.parameters()).dtype)
        proj = self.projector(feats)
        z1, z2 = proj[:bs], proj[bs:]

        inv_loss = F.mse_loss(z1.float(), z2.float())

        z1_global = gather_with_grad(z1.float())
        z2_global = gather_with_grad(z2.float())
        var_loss = 0.5 * (self._variance_loss(z1_global) + self._variance_loss(z2_global))
        cov_loss = 0.5 * (self._covariance_loss(z1_global) + self._covariance_loss(z2_global))

        vicreg_loss = self.sim_coeff * inv_loss + self.var_coeff * var_loss + self.cov_coeff * cov_loss

        prob_loss = vicreg_loss.new_zeros(())
        if get_probe_flag(self.cfg):
            prob_loss = compute_probe_loss(self.probe, feats, y, 2)

        loss = vicreg_loss + prob_loss
        alignment = cosine_alignment(z1, z2)
        collapse_metrics = compute_collapse_metrics(torch.cat([z1, z2], dim=0))

        return loss, inv_loss, var_loss, cov_loss, prob_loss, alignment, collapse_metrics

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

    def _variance_loss(self, x):
        std = torch.sqrt(x.var(dim=0) + self.eps)
        return torch.relu(self.std_target - std).mean()

    def _covariance_loss(self, x):
        x = x - x.mean(dim=0)
        num_samples = max(x.shape[0] - 1, 1)
        cov = (x.T @ x) / num_samples
        return off_diagonal(cov).pow(2).sum() / x.shape[1]
