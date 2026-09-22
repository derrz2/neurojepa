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
    get_method_cfg,
    get_probe_flag,
    resolve_ssl_input,
)
from ..utils.optim import fuse_params_groups, get_params_groups_with_decay


@torch.no_grad()
def distributed_sinkhorn(logits, epsilon, num_iters):
    if epsilon <= 0.0:
        raise ValueError(f"SwAV sinkhorn epsilon must be > 0, got {epsilon}")

    assignments = torch.exp(logits.float() / float(epsilon)).t()
    num_prototypes, local_batch = assignments.shape
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    global_batch = local_batch * world_size

    total_mass = assignments.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total_mass)
    assignments = assignments / total_mass.clamp_min(1e-12)

    for _ in range(int(num_iters)):
        row_sums = assignments.sum(dim=1, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(row_sums)
        assignments = assignments / row_sums.clamp_min(1e-12)
        assignments = assignments / float(num_prototypes)

        col_sums = assignments.sum(dim=0, keepdim=True)
        assignments = assignments / col_sums.clamp_min(1e-12)
        assignments = assignments / float(global_batch)

    assignments = assignments * float(global_batch)
    return assignments.t()


class SwAVModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.cfg = args
        self.swav_cfg = get_method_cfg(args, "swav")

        self.backbone = build_backbone(args, downstream=False)
        embed_dim = int(self.backbone.embed_dim)

        proj_hidden_dim = int(getattr(self.swav_cfg, "proj_hidden_dim", 2048))
        proj_output_dim = int(getattr(self.swav_cfg, "proj_output_dim", 128))
        proj_layers = int(getattr(self.swav_cfg, "proj_num_layers", 2))
        self.num_prototypes = int(getattr(self.swav_cfg, "num_prototypes", 256))
        self.temperature = float(getattr(self.swav_cfg, "temperature", 0.1))
        self.sinkhorn_epsilon = float(getattr(self.swav_cfg, "sinkhorn_epsilon", 0.05))
        self.sinkhorn_iters = int(getattr(self.swav_cfg, "sinkhorn_iters", 3))
        self.n_global_views = int(getattr(self.swav_cfg, "n_global_views", 2))
        self.n_local_views = int(getattr(self.swav_cfg, "n_local_views", 0))

        if self.temperature <= 0.0:
            raise ValueError(f"SwAV temperature must be > 0, got {self.temperature}")
        if self.num_prototypes <= 0:
            raise ValueError(f"SwAV num_prototypes must be > 0, got {self.num_prototypes}")

        self.projector = build_mlp_head(embed_dim, proj_hidden_dim, proj_output_dim, proj_layers)
        self.prototypes = nn.Linear(proj_output_dim, self.num_prototypes, bias=False)

        if get_probe_flag(args):
            self.probe = build_probe(embed_dim)

    @torch.no_grad()
    def _normalize_prototypes(self):
        weight = self.prototypes.weight.data
        self.prototypes.weight.copy_(F.normalize(weight, dim=1))

    def _encode_views(self, views, n_anchor_views):
        if len(views) == n_anchor_views:
            feats = self.backbone(torch.cat(views, dim=0))
            return feats, list(feats.chunk(len(views)))

        anchor_feats = self.backbone(torch.cat(views[:n_anchor_views], dim=0))
        local_feats = self.backbone(torch.cat(views[n_anchor_views:], dim=0))
        all_feats = torch.cat([anchor_feats, local_feats], dim=0)
        view_feats = list(anchor_feats.chunk(n_anchor_views))
        view_feats.extend(local_feats.chunk(len(views) - n_anchor_views))
        return all_feats, view_feats

    def _compute_alignment(self, anchor_proj):
        if len(anchor_proj) == 0:
            return self.prototypes.weight.new_zeros(())
        if len(anchor_proj) == 1:
            return anchor_proj[0].new_zeros(())
        center = torch.stack(anchor_proj, dim=0).mean(dim=0)
        return torch.stack([cosine_alignment(view, center) for view in anchor_proj], dim=0).mean()

    def forward(self, input_batch):
        min_views = max(int(self.n_global_views) + int(self.n_local_views), 2)
        views, y = resolve_ssl_input(input_batch, probe_val=get_probe_flag(self.cfg), min_views=min_views)
        n_views = len(views)
        n_anchor_views = min(self.n_global_views, n_views)
        if n_anchor_views <= 0:
            raise ValueError("SwAV expects at least one anchor view.")

        feats, view_feats = self._encode_views(views, n_anchor_views=n_anchor_views)
        feats = feats.to(next(self.projector.parameters()).dtype)
        proj = F.normalize(self.projector(feats).float(), dim=-1)
        view_proj = list(proj.chunk(n_views))

        self._normalize_prototypes()
        logits_list = [self.prototypes(view) for view in view_proj]

        loss_terms = []
        with torch.no_grad():
            assignments = [
                distributed_sinkhorn(logits_list[i].detach(), self.sinkhorn_epsilon, self.sinkhorn_iters)
                for i in range(n_anchor_views)
            ]

        for i in range(n_anchor_views):
            target = assignments[i]
            for j in range(n_views):
                if j == i:
                    continue
                log_probs = F.log_softmax(logits_list[j] / self.temperature, dim=-1)
                loss_terms.append(-(target * log_probs).sum(dim=-1).mean())

        if len(loss_terms) == 0:
            swav_loss = proj.new_zeros(())
        else:
            swav_loss = torch.stack(loss_terms, dim=0).mean()

        probe_loss = swav_loss.new_zeros(())
        if get_probe_flag(self.cfg):
            anchor_feats = torch.cat(view_feats[:n_anchor_views], dim=0)
            probe_loss = compute_probe_loss(self.probe, anchor_feats, y, n_anchor_views)

        loss = swav_loss + probe_loss
        alignment = self._compute_alignment(view_proj[:n_anchor_views])
        collapse_metrics = compute_collapse_metrics(proj)

        return loss, swav_loss, probe_loss, alignment, collapse_metrics

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
