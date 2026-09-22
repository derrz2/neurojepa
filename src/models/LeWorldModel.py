import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .lejepa.lejepa import multivariate, univariate
from .ssl_utils import (
    build_mlp_head,
    build_probe,
    compute_collapse_metrics,
    compute_probe_loss,
    cosine_alignment,
    get_method_cfg,
    get_probe_flag,
)
from ..utils.optim import fuse_params_groups, get_params_groups_with_decay


class TemporalCausalPredictor(nn.Module):
    def __init__(
        self,
        *,
        input_dim,
        model_dim,
        output_dim,
        history_size,
        depth,
        num_heads,
        mlp_ratio,
        dropout,
    ):
        super().__init__()
        self.history_size = int(history_size)
        self.input_proj = nn.Linear(int(input_dim), int(model_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.history_size, int(model_dim)))

        layer = nn.TransformerEncoderLayer(
            d_model=int(model_dim),
            nhead=int(num_heads),
            dim_feedforward=int(round(float(model_dim) * float(mlp_ratio))),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(depth))
        self.norm = nn.LayerNorm(int(model_dim))
        self.pred = nn.Linear(int(model_dim), int(output_dim))

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected (B, H, D), got {tuple(x.shape)}")
        if x.shape[1] != self.history_size:
            raise ValueError(f"Expected history size {self.history_size}, got {x.shape[1]}")

        x = self.input_proj(x) + self.pos_embed[:, : x.shape[1]]
        mask = torch.full((x.shape[1], x.shape[1]), float("-inf"), device=x.device, dtype=x.dtype)
        mask = torch.triu(mask, diagonal=1)
        x = self.encoder(x, mask=mask)
        x = self.norm(x[:, -1])
        return self.pred(x)


class LeWorldModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.cfg = args
        self.leworld_cfg = get_method_cfg(args, "leworld")

        self.backbone = build_backbone(args, downstream=False)
        backbone_dim = int(self.backbone.embed_dim)

        self.state_len = int(getattr(self.leworld_cfg, "state_len", 40))
        self.state_stride = int(getattr(self.leworld_cfg, "state_stride", 20))
        self.history_size = int(getattr(self.leworld_cfg, "history_size", 3))
        self.horizon = int(getattr(self.leworld_cfg, "horizon", 1))
        self.predict_delta = bool(getattr(self.leworld_cfg, "predict_delta", False))
        self.normalize_targets = bool(getattr(self.leworld_cfg, "normalize_targets", True))
        self.loss_type = str(getattr(self.leworld_cfg, "loss_type", "mse")).lower()

        self.prediction_level = str(getattr(self.leworld_cfg, "prediction_level", "pooled")).lower()
        if self.prediction_level not in {"pooled", "token", "hybrid"}:
            raise ValueError(f"Unsupported prediction_level: {self.prediction_level}")
        self.use_token = self.prediction_level in {"token", "hybrid"}
        self.use_pooled = self.prediction_level in {"pooled", "hybrid"}
        self.pooled_pred_weight = float(getattr(self.leworld_cfg, "pooled_pred_weight", 1.0))
        self.token_pred_weight = float(getattr(self.leworld_cfg, "token_pred_weight", 1.0))

        proj_dim = int(getattr(self.leworld_cfg, "proj_dim", 256))
        proj_hidden_dim = int(getattr(self.leworld_cfg, "proj_hidden_dim", 2048))
        proj_layers = int(getattr(self.leworld_cfg, "proj_layers", 3))
        self.projector = build_mlp_head(
            backbone_dim,
            proj_hidden_dim,
            proj_dim,
            num_layers=proj_layers,
            norm_layer=nn.BatchNorm1d,
        )
        self.predictor = TemporalCausalPredictor(
            input_dim=proj_dim,
            model_dim=int(getattr(self.leworld_cfg, "predictor_embed_dim", 384)),
            output_dim=proj_dim,
            history_size=self.history_size,
            depth=int(getattr(self.leworld_cfg, "predictor_depth", 4)),
            num_heads=int(getattr(self.leworld_cfg, "predictor_num_heads", 6)),
            mlp_ratio=float(getattr(self.leworld_cfg, "predictor_mlp_ratio", 4.0)),
            dropout=float(getattr(self.leworld_cfg, "predictor_dropout", 0.0)),
        )

        if self.use_token:
            token_proj_dim = int(getattr(self.leworld_cfg, "token_proj_dim", proj_dim))
            token_proj_hidden_dim = int(getattr(self.leworld_cfg, "token_proj_hidden_dim", proj_hidden_dim))
            token_proj_layers = int(getattr(self.leworld_cfg, "token_proj_layers", proj_layers))
            self.token_projector = build_mlp_head(
                backbone_dim,
                token_proj_hidden_dim,
                token_proj_dim,
                num_layers=token_proj_layers,
                norm_layer=nn.BatchNorm1d,
            )
            self.token_predictor = TemporalCausalPredictor(
                input_dim=token_proj_dim,
                model_dim=int(getattr(self.leworld_cfg, "token_predictor_embed_dim", 384)),
                output_dim=token_proj_dim,
                history_size=self.history_size,
                depth=int(getattr(self.leworld_cfg, "token_predictor_depth", 4)),
                num_heads=int(getattr(self.leworld_cfg, "token_predictor_num_heads", 6)),
                mlp_ratio=float(getattr(self.leworld_cfg, "token_predictor_mlp_ratio", 4.0)),
                dropout=float(getattr(self.leworld_cfg, "token_predictor_dropout", 0.0)),
            )

        self.use_sigreg = bool(getattr(self.leworld_cfg, "sigreg_enable", True))
        self.sigreg_weight = float(getattr(self.leworld_cfg, "sigreg_weight", 0.05))
        if self.use_sigreg:
            univariate_test = univariate.EppsPulley(
                n_points=int(getattr(self.leworld_cfg, "n_points", 17))
            )
            self.sigreg_loss = multivariate.SlicingUnivariateTest(
                univariate_test=univariate_test,
                num_slices=int(getattr(self.leworld_cfg, "num_slices", 2048)),
            )

        if get_probe_flag(args):
            self.probe = build_probe(backbone_dim)

    def _apply_head(self, head, x):
        shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        x = head(x)
        return x.view(*shape, -1)

    def _resolve_input(self, input_batch):
        x, y = input_batch if get_probe_flag(self.cfg) else (input_batch, None)
        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                raise ValueError("Expected at least one view for LeWorldModel.")
            x = x[0]
        if x.ndim != 3:
            raise ValueError(f"Expected LeWorldModel input with shape (B, C, T), got {tuple(x.shape)}")
        return x, y

    def _segment_states(self, x):
        if x.shape[-1] < self.state_len:
            raise ValueError(f"Input time dimension {x.shape[-1]} is smaller than state_len {self.state_len}")

        states = x.unfold(dimension=-1, size=self.state_len, step=self.state_stride)
        states = states.permute(0, 2, 1, 3).contiguous()
        min_states = self.history_size + self.horizon
        if states.shape[1] < min_states:
            raise ValueError(
                f"Need at least {min_states} temporal states, got {states.shape[1]}. "
                f"Check state_len={self.state_len} and state_stride={self.state_stride}."
            )
        return states

    def _encode_states(self, states):
        b, s, c, t = states.shape
        flat_states = states.view(b * s, c, t)
        state_emb = self.backbone(flat_states)
        state_emb = state_emb.to(next(self.projector.parameters()).dtype)
        state_latent = self._apply_head(self.projector, state_emb)
        return state_emb.view(b, s, -1), state_latent.view(b, s, -1)

    def _encode_tokens(self, states):
        b, s, c, t = states.shape
        flat_states = states.view(b * s, c, t)
        token_emb = self.backbone(flat_states, return_tokens=True)
        token_emb = token_emb.to(next(self.token_projector.parameters()).dtype)
        token_latent = self._apply_head(self.token_projector, token_emb)
        return token_latent.view(b, s, token_latent.shape[1], -1)

    def _build_pairs(self, z):
        b, s, d = z.shape
        n_pairs = s - self.history_size - self.horizon + 1
        ctx, tgt, anc = [], [], []
        for start in range(n_pairs):
            end = start + self.history_size
            target_idx = end + self.horizon - 1
            ctx.append(z[:, start:end])
            tgt.append(z[:, target_idx])
            anc.append(z[:, end - 1])
        ctx = torch.stack(ctx, dim=1).reshape(b * n_pairs, self.history_size, d)
        tgt = torch.stack(tgt, dim=1).reshape(b * n_pairs, d)
        anc = torch.stack(anc, dim=1).reshape(b * n_pairs, d)
        return ctx, tgt, anc

    def _build_token_pairs(self, z):
        b, s, p, d = z.shape
        n_pairs = s - self.history_size - self.horizon + 1
        ctx, tgt, anc = [], [], []
        for start in range(n_pairs):
            end = start + self.history_size
            target_idx = end + self.horizon - 1
            ctx.append(z[:, start:end])
            tgt.append(z[:, target_idx])
            anc.append(z[:, end - 1])

        ctx = torch.stack(ctx, dim=1).permute(0, 1, 3, 2, 4).contiguous()
        tgt = torch.stack(tgt, dim=1).contiguous()
        anc = torch.stack(anc, dim=1).contiguous()
        ctx = ctx.view(b * n_pairs * p, self.history_size, d)
        tgt = tgt.view(b * n_pairs * p, d)
        anc = anc.view(b * n_pairs * p, d)
        return ctx, tgt, anc

    def _normalize_latent(self, x):
        if not self.normalize_targets:
            return x.float()
        return F.layer_norm(x.float(), (x.shape[-1],))

    def _prediction_loss(self, pred, target):
        pred = self._normalize_latent(pred)
        target = self._normalize_latent(target)
        if self.loss_type == "mse":
            return F.mse_loss(pred, target)
        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred, target)
        raise ValueError(f"Unsupported loss_type: {self.loss_type}")

    def _target_from_anchor(self, target, anchor):
        if self.predict_delta:
            return target - anchor
        return target

    def forward(self, input_batch):
        x, y = self._resolve_input(input_batch)
        states = self._segment_states(x)
        state_emb, state_latent = self._encode_states(states)

        pred_loss = state_latent.new_zeros(())
        pooled_pred_loss = state_latent.new_zeros(())
        token_pred_loss = state_latent.new_zeros(())
        alignments = []

        if self.use_pooled:
            ctx, tgt, anc = self._build_pairs(state_latent)
            pred = self.predictor(ctx)
            tgt = self._target_from_anchor(tgt, anc)
            pooled_pred_loss = self._prediction_loss(pred, tgt)
            pred_loss = pred_loss + self.pooled_pred_weight * pooled_pred_loss
            alignments.append(cosine_alignment(pred, tgt))

        if self.use_token:
            token_latent = self._encode_tokens(states)
            ctx, tgt, anc = self._build_token_pairs(token_latent)
            pred = self.token_predictor(ctx)
            tgt = self._target_from_anchor(tgt, anc)
            token_pred_loss = self._prediction_loss(pred, tgt)
            pred_loss = pred_loss + self.token_pred_weight * token_pred_loss
            alignments.append(cosine_alignment(pred, tgt))

        sigreg_loss = pred_loss.new_zeros(())
        if self.use_sigreg:
            sigreg_loss = self.sigreg_loss(state_latent.reshape(-1, state_latent.shape[-1]).float()).mean()
        loss = pred_loss + self.sigreg_weight * sigreg_loss

        probe_loss = pred_loss.new_zeros(())
        if get_probe_flag(self.cfg):
            probe_loss = compute_probe_loss(self.probe, state_emb.mean(dim=1), y, 1)
            loss = loss + probe_loss

        alignment = torch.stack(alignments).mean() if len(alignments) > 0 else pred_loss.new_zeros(())
        collapse_metrics = compute_collapse_metrics(state_latent.reshape(-1, state_latent.shape[-1]))
        return (
            loss,
            pred_loss,
            sigreg_loss,
            probe_loss,
            alignment,
            collapse_metrics,
            pooled_pred_loss,
            token_pred_loss,
        )

    @torch.no_grad()
    def extract_probe_features(self, x):
        if isinstance(x, (list, tuple)):
            x = x[0]
        states = self._segment_states(x)
        state_emb, _ = self._encode_states(states)
        return state_emb.mean(dim=1)

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
