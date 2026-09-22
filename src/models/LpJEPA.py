import contextlib

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .layer.lpjepa_proj import Projections
from ..loss.rectified_lpjepa_loss import (
    rectified_lp_jepa_loss,
    rectified_lp_jepa_multi_view_loss,
    choose_sigma_for_unit_var,
    determine_sigma_for_lp_dist,
)
from ..utils.optim import get_params_groups_with_decay, fuse_params_groups


class RectifiedLpJEPA(nn.Module):
    """Rectified LpJEPA with two-view and global-local training modes."""

    def __init__(self, args):
        super().__init__()
        self.cfg = args

        self.lpjepa_cfg = getattr(args.model, "lpjepa", None)
        if self.lpjepa_cfg is None:
            self.lpjepa_cfg = getattr(args.model, "lpjepa_cfg")

        self.backbone = build_backbone(args, downstream=False)
        self.view_mode = str(args.data.augment.view_mode)

        self.invariance_loss_weight = float(self.lpjepa_cfg.invariance_loss_weight)
        self.rdm_reg_loss_weight = float(self.lpjepa_cfg.rdm_reg_loss_weight)
        self.target_distribution = str(self.lpjepa_cfg.target_distribution)
        self.num_projections = int(self.lpjepa_cfg.num_projections)
        self.projection_vectors_type = str(self.lpjepa_cfg.projection_vectors_type)
        self.mean_shift_value = float(self.lpjepa_cfg.mean_shift_value)
        self.lp_norm_parameter = float(self.lpjepa_cfg.lp_norm_parameter)
        self.n_global_views = int(getattr(self.lpjepa_cfg, "n_global_views", 2))
        self.n_local_views = int(
            getattr(self.lpjepa_cfg, "n_local_views", getattr(args.data.augment, "local_crops_number", 0))
        )

        mode_of_sigma = str(self.lpjepa_cfg.mode_of_sigma)
        if mode_of_sigma == "sigma_GN":
            self.chosen_sigma = determine_sigma_for_lp_dist(self.lp_norm_parameter)
        elif mode_of_sigma == "sigma_RGN":
            self.chosen_sigma = choose_sigma_for_unit_var(self.lp_norm_parameter, self.mean_shift_value)
        else:
            raise ValueError(f"Invalid mode_of_sigma: {mode_of_sigma}")

        proj_hidden_dim = int(self.lpjepa_cfg.proj_hidden_dim)
        self.proj_output_dim = int(self.lpjepa_cfg.proj_output_dim)
        projector_type = str(self.lpjepa_cfg.projector_type)

        in_dim = int(self.backbone.embed_dim)
        if projector_type == "mlp":
            self.projector = nn.Sequential(
                nn.Linear(in_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, self.proj_output_dim),
            )
        elif projector_type == "rectified_mlp":
            self.projector = nn.Sequential(
                nn.Linear(in_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, self.proj_output_dim),
                nn.ReLU(),
            )
        else:
            raise ValueError(f"Invalid projector_type: {projector_type}")

        if self.cfg.training.probe_val:
            self.probe = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, 2))

    def probe_loss(self, emb, y, n_views):
        probe_param = next(self.probe.parameters())
        probe_input = emb.detach().to(device=probe_param.device, dtype=probe_param.dtype)
        autocast_ctx = (
            torch.autocast(device_type=probe_input.device.type, enabled=False)
            if probe_input.device.type != "cpu"
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            yhat = self.probe(probe_input)
            y_rep = y.repeat(n_views).to(device=yhat.device, dtype=torch.long)
            return F.cross_entropy(yhat.float(), y_rep)

    def forward(self, input):
        x, y = input if self.cfg.training.probe_val else (input, None)

        if self.view_mode == "multi_view":
            if len(x) < 2:
                raise ValueError(f"Expected at least 2 views for multi_view, got {len(x)}.")

            x1, x2 = x[0], x[1]
            bs = x1.shape[0]

            feats = self.backbone(torch.cat([x1, x2], dim=0))
            feats = feats.to(next(self.projector.parameters()).dtype)
            proj = self.projector(feats)
            z1, z2 = proj[:bs], proj[bs:]

            projection_vectors = Projections.get_projection_vectors(
                z1,
                z2,
                self.num_projections,
                self.projection_vectors_type,
                self.proj_output_dim,
            )

            loss, inv_loss, reg_loss = rectified_lp_jepa_loss(
                z1,
                z2,
                projection_vectors,
                target_distribution=self.target_distribution,
                invariance_loss_weight=self.invariance_loss_weight,
                rdm_reg_loss_weight=self.rdm_reg_loss_weight,
                mean_shift_value=self.mean_shift_value,
                lp_norm_parameter=self.lp_norm_parameter,
                chosen_sigma=self.chosen_sigma,
            )

            prob_loss = self.probe_loss(feats, y, 2) if self.cfg.training.probe_val else 0.0
            loss = loss + prob_loss
            collapse_metrics = self._compute_collapse_metrics(torch.cat([z1, z2], dim=0))
            return loss, inv_loss, reg_loss, prob_loss, collapse_metrics

        if self.view_mode != "global_local":
            raise ValueError(f"Unsupported view_mode for RectifiedLpJEPA: {self.view_mode}")

        n_views = len(x)
        n_anchor_views = min(self.n_global_views, n_views)
        if n_anchor_views <= 0:
            raise ValueError("Expected at least one global view in global_local mode.")

        global_views = x[:n_anchor_views]
        anchor_output = self.backbone(torch.cat(global_views, dim=0))

        if n_views > n_anchor_views:
            local_output = self.backbone(torch.cat(x[n_anchor_views:], dim=0))
            feats = torch.cat([anchor_output, local_output], dim=0)
        else:
            feats = anchor_output

        feats = feats.to(next(self.projector.parameters()).dtype)
        proj = self.projector(feats)

        bs = global_views[0].shape[0]
        anchor_proj = proj[: n_anchor_views * bs]
        projection_vectors = Projections.get_shared_projection_vectors(
            anchor_proj,
            self.num_projections,
            self.projection_vectors_type,
            self.proj_output_dim,
        )

        loss, inv_loss, reg_loss = rectified_lp_jepa_multi_view_loss(
            proj,
            n_views=n_views,
            n_anchor_views=n_anchor_views,
            projection_vectors=projection_vectors,
            target_distribution=self.target_distribution,
            invariance_loss_weight=self.invariance_loss_weight,
            rdm_reg_loss_weight=self.rdm_reg_loss_weight,
            mean_shift_value=self.mean_shift_value,
            lp_norm_parameter=self.lp_norm_parameter,
            chosen_sigma=self.chosen_sigma,
        )

        prob_loss = self.probe_loss(anchor_output, y, n_anchor_views) if self.cfg.training.probe_val else 0.0
        loss = loss + prob_loss
        collapse_metrics = self._compute_collapse_metrics(proj)

        return loss, inv_loss, reg_loss, prob_loss, collapse_metrics

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

    @torch.no_grad()
    def _compute_collapse_metrics(self, emb):
        if dist.is_available() and dist.is_initialized():
            gathered_emb = [torch.zeros_like(emb) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_emb, emb)
            global_emb = torch.cat(gathered_emb, dim=0)
        else:
            global_emb = emb

        D = global_emb.shape[1]

        mean_vec = global_emb.mean(dim=0)
        mean_norm = mean_vec.norm()

        var_per_dim = global_emb.var(dim=0)
        std_per_dim = torch.sqrt(var_per_dim)
        mean_std = std_per_dim.mean()
        active_dims = (std_per_dim > 1e-4).sum().float()

        var_norm = var_per_dim / (var_per_dim.sum() + 1e-8)
        entropy = -(var_norm * torch.log(var_norm + 1e-8)).sum()
        effective_rank_ratio = entropy / torch.log(torch.tensor(float(D), device=emb.device))

        return {
            "mean_norm": mean_norm,
            "mean_std": mean_std,
            "active_dims": active_dims,
            "effective_rank_ratio": effective_rank_ratio,
        }
