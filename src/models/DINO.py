import torch
import torch.nn as nn
from functools import partial

from .backbone import build_backbone
from .layer.dino_heads import DINOHead
from .ssl_utils import (
    build_probe,
    compute_probe_loss,
    get_probe_flag,
    get_probe_params_groups,
    resolve_ssl_input,
)
from ..loss.dinocls_loss import DINOLoss
from ..loss.koleo_loss import KoLeoLoss
from ..utils.optim import get_params_groups_with_decay, fuse_params_groups

import logging
logger = logging.getLogger("neurojepa")

class DINO_VIT(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.cfg = args
        s_backbone, t_backbone, self.embed_dim = build_dino_model(args)

        d_head = partial(
                DINOHead,
                in_dim=self.embed_dim,
                out_dim=args.model.head_n_prototypes,
                hidden_dim=args.model.head_hidden_dim,
                bottleneck_dim=args.model.head_bottleneck_dim,
                nlayers=args.model.head_nlayers,
            )

        self.student = nn.ModuleDict({"backbone": s_backbone, "dino_head": d_head()})
        self.teacher = nn.ModuleDict({"backbone": t_backbone, "dino_head": d_head()})

        self.dino_loss = DINOLoss(args.model.head_n_prototypes)

        self.koleo_loss_weight = args.model.koleo_loss_weight
        self.do_koleo = args.model.do_koleo
        if self.do_koleo:
            logger.info("OPTIONS -- DINO -- applying KOLEO regularization")
            self.koleo_loss = KoLeoLoss()

        self.n_global_crops = args.model.n_global_views
        self.n_global_crops_loss_terms = (self.n_global_crops - 1) * self.n_global_crops

        self.n_local_crops = args.model.n_local_views
        self.n_local_crops_loss_terms = self.n_global_crops * self.n_local_crops

        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        if get_probe_flag(args):
            self.probe = build_probe(self.embed_dim)

    @property
    def backbone(self):
        return self.student["backbone"]

    def forward(self, x, teacher_temp, update_center=True):
        y = None
        if get_probe_flag(self.cfg):
            x, y = resolve_ssl_input(
                x,
                probe_val=True,
                min_views=self.n_global_crops + max(self.n_local_crops, 0),
            )

        n_global = self.n_global_crops
        n_local = self.n_local_crops

        g_view = torch.cat(x[:n_global], dim=0)
        if n_local > 0:
            l_view = torch.cat(x[n_global:], dim=0)

        with torch.no_grad():
            t_g_cls = self.teacher.backbone(g_view)
            t_cls = self.teacher.dino_head(t_g_cls)
            t_outputs = self.dino_loss.softmax_center_teacher(t_cls, teacher_temp)
            t_out_chunked = t_outputs.chunk(n_global)

        student_g_cls = self.student.backbone(g_view)

        feats = [student_g_cls]
        if n_local > 0:
            feats.append(self.student.backbone(l_view))

        student_a_output = self.student.dino_head(torch.cat(feats))

        batch_size = t_out_chunked[0].shape[0]
        g_len = student_g_cls.shape[0]
        global_student_out = student_a_output[:g_len]
        local_student_out_list = student_a_output[g_len:].chunk(n_local) if n_local > 0 else []

        with torch.no_grad():
            t_logits_centered = (t_cls - self.dino_loss.center) / teacher_temp
            
            t_probs = t_outputs 
            avg_entropy = -torch.sum(t_probs * torch.log(t_probs + 1e-10), dim=-1).mean()
            
            logits_std = t_logits_centered.std()

            top_indices = t_outputs.chunk(n_global)[0].argmax(dim=-1)
            unique_in_batch = len(torch.unique(top_indices))
            if not hasattr(self, 'diag_step'): self.diag_step = 0
            self.diag_step += 1
            
            if self.diag_step % 50 == 0:
                logger.info(
                    f"[DIAGNOSTIC] Step {self.diag_step} | "
                    f"Unique/Batch: {unique_in_batch}/{batch_size} | "
                    f"Entropy: {avg_entropy:.4f} | "
                    f"LogitsStd: {logits_std:.4f}")

        teacher_global_cross_view = torch.cat([t_out_chunked[1], t_out_chunked[0]], dim=0)

        loss_g_scales = 2
        loss_global = (
            self.dino_loss(
                [global_student_out],         
                [teacher_global_cross_view]    
            )
            * loss_g_scales
            / (self.n_local_crops_loss_terms + self.n_global_crops_loss_terms)
        )

        loss_local = 0.0
        if n_local > 0:
            loss_local = self.dino_loss(
                local_student_out_list, 
                t_out_chunked
            ) / (self.n_local_crops_loss_terms + self.n_global_crops_loss_terms)

        loss_koleo = 0.0
        if self.do_koleo:
            # student_g_cls shape: [Batch_Size * 2, Embed_Dim] -> chunk(2) -> 2 * [Batch_Size, Embed_Dim]
            loss_koleo = sum(self.koleo_loss(p) for p in student_g_cls.chunk(2))
            loss_koleo = loss_koleo * self.koleo_loss_weight

        prob_loss = loss_global.new_zeros(())
        if get_probe_flag(self.cfg):
            prob_loss = compute_probe_loss(self.probe, student_g_cls, y, n_global)

        loss = loss_global + loss_local + loss_koleo + prob_loss

        if update_center:
            self.dino_loss.update_center(t_cls)

        return loss, loss_global, loss_local, loss_koleo, prob_loss, t_outputs

    @torch.no_grad()
    def update_teacher(self, m):
        for k in self.student.keys():
            student_params = list(self.student[k].parameters())
            teacher_params = list(self.teacher[k].parameters())

            torch._foreach_mul_(teacher_params, m)
            torch._foreach_add_(teacher_params, student_params, alpha=1 - m)

    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)

        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups

    def get_params_groups(self):
        all_params_groups = []
        for m in self.student.values():
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        if hasattr(self, "probe"):
            probe_groups = fuse_params_groups(get_probe_params_groups(self.probe))
            for g in probe_groups:
                g["foreach"] = True
            all_params_groups.extend(list(probe_groups))
        return all_params_groups


def build_dino_model(args, only_teacher=False):
    teacher = build_backbone(args, downstream=False)

    if only_teacher:
        return teacher, teacher.embed_dim
    student = build_backbone(args, downstream=False)

    for param_t, param_s in zip(teacher.parameters(), student.parameters()):
        param_t.data.copy_(param_s.data)
        param_t.requires_grad = False 

    embed_dim = student.embed_dim

    return student, teacher, embed_dim
