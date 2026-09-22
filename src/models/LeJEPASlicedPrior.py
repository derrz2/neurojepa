import contextlib

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .layer.vit_components import MLP
from .lejepa.lejepa import univariate, multivariate
from ..loss.sliced_prior_loss import SlicedPriorMatchingLoss
from ..utils.optim import get_params_groups_with_decay, fuse_params_groups


class LEJEPASlicedPrior_VIT(nn.Module):

    def __init__(
            self,
            args,
    ) -> None:
        super().__init__()

        self.cfg = args
        self.lejepa_cfg = getattr(args.model, "lejepa_sliced_prior", None)
        if self.lejepa_cfg is None:
            self.lejepa_cfg = getattr(args.model, "lejepa")

        self.view_mode = args.data.augment.view_mode
        self.backbone = build_backbone(args, downstream=False)

        backbone_dim = int(self.backbone.embed_dim)
        projector_dims = getattr(self.lejepa_cfg, "projector_dims", None)
        if projector_dims is None:
            projector_dims = [2048, 2048, int(self.lejepa_cfg.proj_dim)]
        else:
            projector_dims = [int(dim) for dim in projector_dims]
        self.proj = nn.Sequential(
            MLP(backbone_dim, projector_dims, norm_layer=nn.BatchNorm1d)
        )

        self.use_prior = bool(getattr(self.lejepa_cfg, "prior_enable", True))
        self.prior_weight = float(getattr(self.lejepa_cfg, "regularizer_weight", getattr(self.lejepa_cfg, "lamda", 0.05)))
        self.use_sigreg = bool(getattr(self.lejepa_cfg, "sigreg_enable", False))
        self.sigreg_weight = float(getattr(self.lejepa_cfg, "sigreg_weight", 0.0))
        self.alpha = float(getattr(self.lejepa_cfg, "alpha", 0.5))

        if self.use_prior:
            self.prior_loss = SlicedPriorMatchingLoss(
                loss_type=str(getattr(self.lejepa_cfg, "regularizer_type", "wasserstein")),
                num_slices=int(getattr(self.lejepa_cfg, "num_slices", 2048)),
                feature_source=str(getattr(self.lejepa_cfg, "feature_source", "centers")),
                target_distribution=str(getattr(self.lejepa_cfg, "target_distribution", "student_t")),
                target_df=float(getattr(self.lejepa_cfg, "target_df", 2.0)),
                target_scale=float(getattr(self.lejepa_cfg, "target_scale", 1.0)),
                gather_distributed=bool(getattr(self.lejepa_cfg, "gather_distributed", True)),
                normalize_slices=bool(getattr(self.lejepa_cfg, "normalize_slices", True)),
                wasserstein_p=int(getattr(self.lejepa_cfg, "wasserstein_p", 1)),
                num_bins=int(getattr(self.lejepa_cfg, "num_bins", 8)),
                occupancy_temperature=float(getattr(self.lejepa_cfg, "occupancy_temperature", 0.25)),
                occupancy_powerlaw_exponent=float(getattr(self.lejepa_cfg, "occupancy_powerlaw_exponent", 1.0)),
                occupancy_max_radius=float(getattr(self.lejepa_cfg, "occupancy_max_radius", 3.0)),
                kurtosis_target=float(getattr(self.lejepa_cfg, "kurtosis_target", 6.0)),
                pairwise_max_samples=int(getattr(self.lejepa_cfg, "pairwise_max_samples", 48)),
                variance_floor_weight=float(getattr(self.lejepa_cfg, "variance_floor_weight", 0.0)),
                variance_floor_target=float(getattr(self.lejepa_cfg, "variance_floor_target", 0.5)),
            )

        if self.use_sigreg:
            univariate_test = univariate.EppsPulley(
                n_points=int(getattr(self.lejepa_cfg, "sigreg_n_points", 17))
            )
            self.sigreg_loss = multivariate.SlicingUnivariateTest(
                univariate_test=univariate_test,
                num_slices=int(getattr(self.lejepa_cfg, "num_slices", 2048)),
            )

        self.n_global_views = int(getattr(self.lejepa_cfg, "n_global_views", 2))
        self.ncrops = int(getattr(self.lejepa_cfg, "n_global_views", 2)) + int(getattr(self.lejepa_cfg, "n_local_views", 0))

        if self.cfg.training.probe_val:
            in_dim = int(self.backbone.embed_dim)
            self.probe = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, 2))

    def lejepa_loss(self, proj, backbone_emb, n_views, n_anchor_views):
        proj = proj.float()
        dim = proj.shape[1]

        bs = proj.shape[0] // n_views
        view_embeddings = proj.view(n_views, bs, dim)
        anchor_emb = view_embeddings[:n_anchor_views]
        local_emb = view_embeddings[n_anchor_views:] if n_views > n_anchor_views else None

        centers = anchor_emb.mean(dim=0)
        anchor_diff = anchor_emb - centers.unsqueeze(0)
        all_diff = view_embeddings - centers.unsqueeze(0)
        flat_emb = view_embeddings.reshape(-1, dim)

        anchor_loss = anchor_diff.square().mean()
        local_loss = None
        if local_emb is not None and local_emb.numel() > 0:
            local_loss = (local_emb - centers.unsqueeze(0)).square().mean()
        inv_loss = all_diff.square().mean()

        if self.use_prior:
            # Match the prior on projected features so the regularizer acts on
            # the same representation space as the JEPA invariance objective.
            prior_loss, prior_metrics = self.prior_loss(view_embeddings, n_anchor_views=n_anchor_views)
            weighted_inv_loss = (1 - self.prior_weight) * inv_loss
            weighted_prior_loss = self.prior_weight * prior_loss
            loss = weighted_inv_loss + weighted_prior_loss
            prior_weight_safe = max(float(self.prior_weight), 1e-8)
            aliment_metric = loss / (prior_weight_safe ** self.alpha)
        else:
            prior_loss = inv_loss.new_zeros(())
            weighted_inv_loss = inv_loss
            weighted_prior_loss = inv_loss.new_zeros(())
            loss = inv_loss
            aliment_metric = inv_loss
            prior_metrics = {}

        if self.use_sigreg:
            sigreg_loss = self.sigreg_loss(flat_emb).mean()
            weighted_sigreg_loss = self.sigreg_weight * sigreg_loss
            loss = loss + weighted_sigreg_loss
        else:
            sigreg_loss = inv_loss.new_zeros(())
            weighted_sigreg_loss = inv_loss.new_zeros(())

        total_reg_weight = max(float(self.prior_weight) + float(self.sigreg_weight), 1e-8)
        aliment_metric = loss / (total_reg_weight ** self.alpha)

        collapse_metrics = self._compute_collapse_metrics(flat_emb)
        extra_metrics = {
            "anchor_loss": anchor_loss,
            "all_loss": inv_loss,
            "weighted_inv_loss": weighted_inv_loss,
            "weighted_prior_loss": weighted_prior_loss,
            "prior_loss": prior_loss,
            "weighted_sigreg_loss": weighted_sigreg_loss,
            "sigreg_loss": sigreg_loss,
        }
        extra_metrics.update(prior_metrics)
        if local_loss is not None:
            extra_metrics["local_loss"] = local_loss
        return loss, inv_loss, prior_loss, sigreg_loss, extra_metrics, aliment_metric, collapse_metrics

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

        n_views = len(x)
        if self.view_mode == "multi_view":
            a_output = self.backbone(torch.cat(x, dim=0))
            n_anchor_views = n_views
            anchor_output = a_output
        else:
            n_anchor_views = min(self.n_global_views, n_views)
            if n_anchor_views <= 0:
                raise ValueError("Expected at least one anchor view.")

            g_view = torch.cat(x[:n_anchor_views], dim=0)
            g_output = self.backbone(g_view)

            anchor_output = g_output
            if n_views > n_anchor_views:
                l_views = x[n_anchor_views:]
                l_output = self.backbone(torch.cat(l_views, dim=0))
                a_output = torch.cat([g_output, l_output], dim=0)
            else:
                a_output = g_output

        a_output = a_output.to(next(self.proj.parameters()).dtype)
        a_proj = self.proj(a_output)
        loss, inv_loss, prior_loss, sigreg_loss, extra_metrics, aliment_metric, collapse_metrics = self.lejepa_loss(
            a_proj,
            a_output,
            n_views=n_views,
            n_anchor_views=n_anchor_views
        )

        prob_loss = self.probe_loss(anchor_output, y, n_anchor_views) if self.cfg.training.probe_val else 0.0
        loss += prob_loss

        return loss, inv_loss, prior_loss, sigreg_loss, prob_loss, extra_metrics, aliment_metric, collapse_metrics

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

        n_samples, dim = global_emb.shape

        mean_vec = global_emb.mean(dim=0)
        mean_norm = mean_vec.norm()

        centered = global_emb - mean_vec
        cov = centered.T @ centered / max(n_samples - 1, 1)
        eigvals = torch.linalg.eigvalsh(cov.float()).clamp_min(1e-12)
        eigvals = eigvals / eigvals.sum()
        eff_rank = torch.exp(-(eigvals * eigvals.log()).sum())

        std_per_dim = global_emb.std(dim=0, unbiased=False)
        mean_std = std_per_dim.mean()
        active_dims = (std_per_dim > 1e-4).sum().float()

        return {
            "mean_norm": mean_norm,
            "mean_std": mean_std,
            "active_dims": active_dims,
            "effective_rank_ratio": eff_rank / float(dim),
        }
