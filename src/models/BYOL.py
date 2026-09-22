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
    get_probe_params_groups,
    negative_cosine_similarity,
    resolve_ssl_input,
)
from ..utils.optim import get_params_groups_with_decay, fuse_params_groups


class BYOLModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.cfg = args
        self.byol_cfg = get_method_cfg(args, "byol")

        student_backbone = build_backbone(args, downstream=False)
        teacher_backbone = build_backbone(args, downstream=False)
        embed_dim = int(student_backbone.embed_dim)

        proj_hidden_dim = int(getattr(self.byol_cfg, "proj_hidden_dim", 2048))
        proj_output_dim = int(getattr(self.byol_cfg, "proj_output_dim", 256))
        pred_hidden_dim = int(getattr(self.byol_cfg, "pred_hidden_dim", 1024))
        proj_layers = int(getattr(self.byol_cfg, "proj_num_layers", 3))
        pred_layers = int(getattr(self.byol_cfg, "pred_num_layers", 2))

        student_projector = build_mlp_head(embed_dim, proj_hidden_dim, proj_output_dim, proj_layers)
        teacher_projector = build_mlp_head(embed_dim, proj_hidden_dim, proj_output_dim, proj_layers)
        student_predictor = build_mlp_head(proj_output_dim, pred_hidden_dim, proj_output_dim, pred_layers)

        self.student = nn.ModuleDict(
            {
                "backbone": student_backbone,
                "projector": student_projector,
                "predictor": student_predictor,
            }
        )
        self.teacher = nn.ModuleDict(
            {
                "backbone": teacher_backbone,
                "projector": teacher_projector,
            }
        )

        self.teacher["backbone"].load_state_dict(self.student["backbone"].state_dict())
        self.teacher["projector"].load_state_dict(self.student["projector"].state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False

        if get_probe_flag(args):
            self.probe = build_probe(embed_dim)

    @property
    def backbone(self):
        return self.student["backbone"]

    def forward(self, input_batch):
        views, y = resolve_ssl_input(input_batch, probe_val=get_probe_flag(self.cfg), min_views=2)
        x1, x2 = views[0], views[1]
        bs = x1.shape[0]

        student_feats = self.student["backbone"](torch.cat([x1, x2], dim=0))
        student_feats = student_feats.to(next(self.student["projector"].parameters()).dtype)
        student_proj = self.student["projector"](student_feats)
        student_pred = self.student["predictor"](student_proj)
        p1, p2 = student_pred[:bs], student_pred[bs:]

        with torch.no_grad():
            teacher_feats = self.teacher["backbone"](torch.cat([x1, x2], dim=0))
            teacher_feats = teacher_feats.to(next(self.teacher["projector"].parameters()).dtype)
            teacher_proj = self.teacher["projector"](teacher_feats)
            z1, z2 = teacher_proj[:bs], teacher_proj[bs:]

        loss_12 = negative_cosine_similarity(p1, z2)
        loss_21 = negative_cosine_similarity(p2, z1)
        byol_loss = 0.5 * (loss_12 + loss_21)

        prob_loss = byol_loss.new_zeros(())
        if get_probe_flag(self.cfg):
            prob_loss = compute_probe_loss(self.probe, student_feats, y, 2)

        loss = byol_loss + prob_loss
        alignment = 0.5 * (cosine_alignment(p1, z2) + cosine_alignment(p2, z1))
        collapse_metrics = compute_collapse_metrics(torch.cat([student_proj[:bs], student_proj[bs:]], dim=0))

        return loss, byol_loss, prob_loss, alignment, collapse_metrics

    @torch.no_grad()
    def update_teacher(self, m):
        for key in self.teacher.keys():
            student_params = list(self.student[key].parameters())
            teacher_params = list(self.teacher[key].parameters())
            torch._foreach_mul_(teacher_params, m)
            torch._foreach_add_(teacher_params, student_params, alpha=1 - m)

    def _get_fused_params_for_submodel(self, module):
        params_groups = get_params_groups_with_decay(
            model=module,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        for g in fused_params_groups:
            g["foreach"] = True
        return list(fused_params_groups)

    def get_params_groups(self):
        all_params_groups = []
        for key in self.student.keys():
            all_params_groups.extend(self._get_fused_params_for_submodel(self.student[key]))
        if hasattr(self, "probe"):
            probe_groups = fuse_params_groups(get_probe_params_groups(self.probe))
            for g in probe_groups:
                g["foreach"] = True
            all_params_groups.extend(list(probe_groups))
        return all_params_groups
