import torch
import torch.nn as nn

from .backbone import build_backbone
from .ssl_utils import (
    build_mlp_head,
    build_probe,
    compute_collapse_metrics,
    compute_probe_loss,
    cosine_alignment,
    get_method_cfg,
    get_probe_flag,
    negative_cosine_similarity,
    resolve_ssl_input,
)
from ..utils.optim import fuse_params_groups, get_params_groups_with_decay


class SimSiamModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.cfg = args
        self.simsiam_cfg = get_method_cfg(args, "simsiam")

        self.backbone = build_backbone(args, downstream=False)
        embed_dim = int(self.backbone.embed_dim)

        proj_hidden_dim = int(getattr(self.simsiam_cfg, "proj_hidden_dim", 2048))
        proj_output_dim = int(getattr(self.simsiam_cfg, "proj_output_dim", 2048))
        pred_hidden_dim = int(getattr(self.simsiam_cfg, "pred_hidden_dim", 512))
        proj_layers = int(getattr(self.simsiam_cfg, "proj_num_layers", 3))
        pred_layers = int(getattr(self.simsiam_cfg, "pred_num_layers", 2))

        self.projector = build_mlp_head(embed_dim, proj_hidden_dim, proj_output_dim, proj_layers)
        self.predictor = build_mlp_head(proj_output_dim, pred_hidden_dim, proj_output_dim, pred_layers)

        if get_probe_flag(args):
            self.probe = build_probe(embed_dim)

    def forward(self, input_batch):
        views, y = resolve_ssl_input(input_batch, probe_val=get_probe_flag(self.cfg), min_views=2)
        x1, x2 = views[0], views[1]
        batch_size = x1.shape[0]

        feats = self.backbone(torch.cat([x1, x2], dim=0))
        feats = feats.to(next(self.projector.parameters()).dtype)
        proj = self.projector(feats)
        pred = self.predictor(proj)

        z1, z2 = proj[:batch_size], proj[batch_size:]
        p1, p2 = pred[:batch_size], pred[batch_size:]

        loss_12 = negative_cosine_similarity(p1, z2.detach())
        loss_21 = negative_cosine_similarity(p2, z1.detach())
        simsiam_loss = 0.5 * (loss_12 + loss_21)

        probe_loss = simsiam_loss.new_zeros(())
        if get_probe_flag(self.cfg):
            probe_loss = compute_probe_loss(self.probe, feats, y, 2)

        loss = simsiam_loss + probe_loss
        alignment = 0.5 * (cosine_alignment(p1, z2) + cosine_alignment(p2, z1))
        collapse_metrics = compute_collapse_metrics(torch.cat([z1, z2], dim=0))

        return loss, simsiam_loss, probe_loss, alignment, collapse_metrics

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
