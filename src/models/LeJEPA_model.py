import contextlib

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .layer.vit_components import MLP
from .lejepa.lejepa import univariate, multivariate

from ..utils.optim import get_params_groups_with_decay, fuse_params_groups

class LEJEPA_VIT(nn.Module):

    def __init__(
            self,
            args,
    ) -> None:
        super().__init__()

        self.cfg = args
        self.lejepa_cfg = args.model.lejepa
        self.use_sigreg = self.lejepa_cfg.sigreg_enable
        self.view_mode = args.data.augment.view_mode
        self.backbone = build_backbone(args, downstream=False)

        self.use_sigreg = self.lejepa_cfg.sigreg_enable
        backbone_dim = int(self.backbone.embed_dim)
        projector_dims = getattr(self.lejepa_cfg, "projector_dims", None)
        if projector_dims is None:
            projector_dims = [2048, 2048, int(self.lejepa_cfg.proj_dim)]
        else:
            projector_dims = [int(dim) for dim in projector_dims]
        self.proj = nn.Sequential(
                MLP(backbone_dim, projector_dims, norm_layer=nn.SyncBatchNorm)
            )
        if self.use_sigreg:
            univariate_test = univariate.EppsPulley(n_points=self.lejepa_cfg.n_points)
            self.sigreg_loss = multivariate.SlicingUnivariateTest(
                univariate_test=univariate_test,
                num_slices=self.lejepa_cfg.num_slices
            )
            self.lamda = self.lejepa_cfg.lamda
            self.alpha = self.lejepa_cfg.alpha

        self.n_global_views = self.lejepa_cfg.n_global_views
        self.ncrops = self.lejepa_cfg.n_global_views + self.lejepa_cfg.n_local_views

        if self.cfg.training.probe_val:
            in_dim = int(self.backbone.embed_dim)
            self.probe = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, 2))

    def lejepa_loss(self, proj, n_views, n_anchor_views):
        proj = proj.float()
        K = proj.shape[1]

        bs = proj.shape[0] // n_views
        v_emb = proj.view(n_views, bs, K)
        anchor_emb = v_emb[:n_anchor_views]
        local_emb = v_emb[n_anchor_views:] if n_views > n_anchor_views else None

        centers = anchor_emb.mean(dim=0)
        anchor_diff = anchor_emb - centers.unsqueeze(0)
        all_diff = v_emb - centers.unsqueeze(0)
        flat_emb = v_emb.reshape(-1, K)

        anchor_loss = anchor_diff.square().mean()
        local_loss = None
        if local_emb is not None and local_emb.numel() > 0:
            local_loss = (local_emb - centers.unsqueeze(0)).square().mean()
        inv_loss = all_diff.square().mean()
        if self.use_sigreg:
            sigreg_loss = self.sigreg_loss(flat_emb).mean()
            weighted_inv_loss = (1 - self.lamda) * inv_loss
            weighted_sigreg_loss = self.lamda * sigreg_loss
            loss = weighted_inv_loss + weighted_sigreg_loss
            lamda_safe = max(float(self.lamda), 1e-8)
            aliment_metric = loss / (lamda_safe ** self.alpha)
        else:
            sigreg_loss = inv_loss.new_zeros(())
            weighted_inv_loss = inv_loss
            weighted_sigreg_loss = inv_loss.new_zeros(())
            loss = inv_loss
            aliment_metric = inv_loss
        collapse_metrics = self._compute_collapse_metrics(flat_emb)
        extra_metrics = {
            "anchor_loss": anchor_loss,
            "all_loss": inv_loss,
            "weighted_inv_loss": weighted_inv_loss,
            "weighted_sigreg_loss": weighted_sigreg_loss,
        }
        if local_loss is not None:
            extra_metrics["local_loss"] = local_loss
        return loss, inv_loss, sigreg_loss, extra_metrics, aliment_metric, collapse_metrics

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

    @staticmethod
    def _unpack_view(view):
        if isinstance(view, dict):
            return view["x"], view.get("patch_valid_mask")
        return view, None

    def _encode_views(self, views):
        x_list = []
        valid_mask_list = []
        for view in views:
            x, patch_valid_mask = self._unpack_view(view)
            x_list.append(x)
            valid_mask_list.append(patch_valid_mask)

        batch = torch.cat(x_list, dim=0)
        if all(mask is None for mask in valid_mask_list):
            return self.backbone(batch), x_list[0].shape[0]
        if any(mask is None for mask in valid_mask_list):
            raise ValueError("Mixed masked and unmasked views are not supported in LeJEPA pretraining.")

        key_padding_mask = torch.cat([~mask for mask in valid_mask_list], dim=0)
        return self.backbone(batch, key_padding_mask=key_padding_mask), x_list[0].shape[0]

    def forward(self, input):
        x, y = input if self.cfg.training.probe_val else (input, None)
        if not isinstance(x, (list, tuple)):
            x = [x]

        n_views = len(x)
        if self.view_mode == "multi_view":
            a_output, _ = self._encode_views(x)
            n_anchor_views = n_views
            anchor_output = a_output
        else:
            g_output, _ = self._encode_views(x[:self.n_global_views])

            l_views = x[self.n_global_views:]
            if len(l_views) > 0:
                l_output, _ = self._encode_views(l_views)
                a_output = torch.cat([g_output, l_output], dim=0)
            else:
                a_output = g_output

            n_anchor_views = self.n_global_views
            anchor_output = g_output
        
        a_output = a_output.to(next(self.proj.parameters()).dtype)
        a_proj = self.proj(a_output)
        loss, inv_loss, sigreg_loss, extra_metrics, aliment_metric, collapse_metrics = self.lejepa_loss(
            a_proj,
            n_views=n_views,
            n_anchor_views=n_anchor_views
        )

        prob_loss = self.probe_loss(anchor_output, y, n_anchor_views) if self.cfg.training.probe_val else 0.0
        loss += prob_loss

        return loss, inv_loss, sigreg_loss, prob_loss, extra_metrics, aliment_metric, collapse_metrics

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

        N, D = global_emb.shape

        mean_vec = global_emb.mean(dim=0)
        mean_norm = mean_vec.norm()

        var_per_dim = global_emb.var(dim=0)
        std_per_dim = torch.sqrt(var_per_dim)
        mean_std = std_per_dim.mean()

        active_dims = (std_per_dim > 1e-4).sum().float()

        var_norm = var_per_dim / (var_per_dim.sum() + 1e-8)
        entropy = - (var_norm * torch.log(var_norm + 1e-8)).sum()
        effective_rank_ratio = entropy / torch.log(torch.tensor(float(D), device=emb.device))

        return {
            "mean_norm": mean_norm,
            "mean_std": mean_std,
            "active_dims": active_dims,
            "effective_rank_ratio": effective_rank_ratio
        }
