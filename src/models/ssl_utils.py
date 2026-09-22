import contextlib

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .layer.vit_components import MLP
from .backbone.base import cfg_get
from ..utils.utils import gather


def get_method_cfg(args, method_name):
    model_cfg = cfg_get(args, "model", args)
    return cfg_get(model_cfg, method_name, {})


def get_probe_flag(args) -> bool:
    train_cfg = cfg_get(args, "training", {})
    return bool(cfg_get(train_cfg, "probe_val", False))


def resolve_ssl_input(input_batch, *, probe_val=False, min_views=2):
    if probe_val:
        views, labels = input_batch
    else:
        views, labels = input_batch, None

    if not isinstance(views, (list, tuple)):
        raise ValueError(
            f"Expected SSL input to be a list/tuple of views, got {type(views).__name__}."
        )
    if len(views) < min_views:
        raise ValueError(f"Expected at least {min_views} views, got {len(views)}.")
    return list(views[:min_views]), labels


def build_mlp_head(in_dim, hidden_dim, out_dim, num_layers=3, norm_layer=nn.BatchNorm1d):
    num_layers = max(int(num_layers), 1)
    if num_layers == 1:
        return nn.Linear(int(in_dim), int(out_dim))
    hidden_dims = [int(hidden_dim)] * (num_layers - 1) + [int(out_dim)]
    return MLP(int(in_dim), hidden_dims, norm_layer=norm_layer, act_layer=nn.ReLU)


def build_probe(in_dim, num_classes=2):
    return nn.Sequential(nn.LayerNorm(int(in_dim)), nn.Linear(int(in_dim), int(num_classes)))


def compute_probe_loss(probe, emb, y, n_views):
    probe_param = next(probe.parameters())
    probe_input = emb.detach().to(device=probe_param.device, dtype=probe_param.dtype)
    autocast_ctx = (
        torch.autocast(device_type=probe_input.device.type, enabled=False)
        if probe_input.device.type != "cpu"
        else contextlib.nullcontext()
    )
    with autocast_ctx:
        yhat = probe(probe_input)
        y_rep = y.repeat(n_views).to(device=yhat.device, dtype=torch.long)
        return F.cross_entropy(yhat.float(), y_rep)


@torch.no_grad()
def compute_collapse_metrics(emb):
    emb = emb.detach().float().contiguous()
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
    std_per_dim = torch.sqrt(var_per_dim + 1e-8)
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


def cosine_alignment(x, y):
    x = F.normalize(x.float(), dim=-1)
    y = F.normalize(y.float(), dim=-1)
    return F.cosine_similarity(x, y, dim=-1).mean()


def negative_cosine_similarity(p, z):
    p = F.normalize(p.float(), dim=-1)
    z = F.normalize(z.float(), dim=-1)
    return 2.0 - 2.0 * (p * z).sum(dim=-1).mean()


def gather_with_grad(x):
    if dist.is_available() and dist.is_initialized():
        return gather(x)
    return x


def off_diagonal(x):
    n, m = x.shape
    if n != m:
        raise ValueError(f"Expected square matrix, got {tuple(x.shape)}")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def get_probe_params_groups(probe):
    params_groups = []
    for name, param in probe.named_parameters():
        if not param.requires_grad:
            continue
        wd_multiplier = 0.0 if (name.endswith(".bias") or param.ndim <= 1 or "norm" in name.lower()) else 1.0
        params_groups.append(
            {
                "params": param,
                "is_last_layer": False,
                "is_probe": True,
                "lr_multiplier": 1.0,
                "wd_multiplier": wd_multiplier,
                "name": f"probe.{name}",
            }
        )
    return params_groups
