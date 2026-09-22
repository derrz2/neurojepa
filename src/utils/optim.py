import numpy as np
import logging
from collections import defaultdict
logger = logging.getLogger("neurojepa")

class CosineScheduler(object):
    def __init__(self, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, freeze_iters=0):
        super().__init__()
        total_iters = int(total_iters)
        warmup_iters = max(0, min(int(warmup_iters), total_iters))
        freeze_iters = max(0, min(int(freeze_iters), total_iters - warmup_iters))
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros((freeze_iters))

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        remaining_iters = total_iters - warmup_iters - freeze_iters
        if remaining_iters > 0:
            iters = np.arange(remaining_iters)
            schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        else:
            schedule = np.array([], dtype=np.float64)
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))

        assert len(self.schedule) == self.total_iters

    def __len__(self):
        return self.total_iters

    def __getitem__(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]


class WarmupThenConstantScheduler(object):
    def __init__(self, base_value, final_value=None, total_iters=0, warmup_iters=0, start_warmup_value=0, freeze_iters=0):
        super().__init__()
        total_iters = int(total_iters)
        warmup_iters = max(0, min(int(warmup_iters), total_iters))
        freeze_iters = max(0, min(int(freeze_iters), total_iters - warmup_iters))
        self.final_value = base_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros((freeze_iters))
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        remaining_iters = total_iters - warmup_iters - freeze_iters
        if remaining_iters > 0:
            schedule = np.full((remaining_iters), float(base_value), dtype=np.float64)
        else:
            schedule = np.array([], dtype=np.float64)
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))

        assert len(self.schedule) == self.total_iters

    def __len__(self):
        return self.total_iters

    def __getitem__(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]

def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr, probe_lr=None, probe_wd=None):
    for param_group in optimizer.param_groups:
        is_probe = param_group.get("is_probe", False)
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        if is_probe:
            base_lr = probe_lr if probe_lr is not None else lr
            base_wd = probe_wd if probe_wd is not None else wd
        else:
            base_lr = last_layer_lr if is_last_layer else lr
            base_wd = wd
        param_group["weight_decay"] = base_wd * wd_multiplier
        param_group["lr"] = base_lr * lr_multiplier


def _resolve_warmup_iters(
    total_train_iters,
    mode="epochs",
    warmup_epochs=0,
    official_epoch_length=1,
    warmup_ratio=None,
    warmup_min_steps=0,
    warmup_max_steps=None,
):
    total_train_iters = int(total_train_iters)
    mode = (mode or "epochs").lower()

    if mode == "step_ratio":
        ratio = float(0.0 if warmup_ratio is None else warmup_ratio)
        warmup_iters = int(round(total_train_iters * ratio))
        warmup_iters = max(int(warmup_min_steps or 0), warmup_iters)
        if warmup_max_steps is not None:
            warmup_iters = min(warmup_iters, int(warmup_max_steps))
    elif mode == "epochs":
        warmup_iters = int(warmup_epochs) * int(official_epoch_length)
    else:
        raise ValueError(f"Unsupported warmup mode: {mode}")

    return max(0, min(int(warmup_iters), total_train_iters))


def build_schedulers(config):
    """Create learning rate scheduler"""
    teacher_cfg = config.get('teacher', {})
    OFFICIAL_EPOCH_LENGTH = config['training']['OFFICIAL_EPOCH_LENGTH']
    total_train_iters = config.optim["epochs"] * OFFICIAL_EPOCH_LENGTH
    max_steps = config['training'].get('max_steps', None)
    if max_steps is not None:
        total_train_iters = min(total_train_iters, int(max_steps))

    lr_warmup_iters = _resolve_warmup_iters(
        total_train_iters=total_train_iters,
        mode=config.optim.get("warmup_mode", "epochs"),
        warmup_epochs=config.optim.get("warmup_epochs", 0),
        official_epoch_length=OFFICIAL_EPOCH_LENGTH,
        warmup_ratio=config.optim.get("warmup_ratio", None),
        warmup_min_steps=config.optim.get("warmup_min_steps", 0),
        warmup_max_steps=config.optim.get("warmup_max_steps", None),
    )

    lr = dict(
        base_value=config.optim["lr"],
        final_value=config.optim["min_lr"],
        total_iters=total_train_iters,
        warmup_iters=lr_warmup_iters,
        start_warmup_value=0,
    )
    wd = dict(
        base_value=config.optim["weight_decay"],
        final_value=config.optim["weight_decay_end"],
        total_iters=total_train_iters,
    )

    lr_schedule_type = str(config.optim.get("lr_schedule", "cosine")).lower()
    if lr_schedule_type in ("cosine", "cos"):
        lr_scheduler_cls = CosineScheduler
    elif lr_schedule_type in ("constant_after_warmup", "warmup_constant", "constant"):
        lr_scheduler_cls = WarmupThenConstantScheduler
    else:
        raise ValueError(f"Unsupported optim.lr_schedule: {lr_schedule_type}")

    lr_schedule = lr_scheduler_cls(**lr)
    wd_schedule = CosineScheduler(**wd)

    last_layer_lr_schedule = lr_scheduler_cls(**lr)

    last_layer_lr_schedule.schedule[
        : config.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
    ] = 0  

    t_temp_base = teacher_cfg.get("teacher_temp") 
        
    t_temp_warmup = teacher_cfg.get("warmup_teacher_temp")
        
    t_temp_warmup_epochs = teacher_cfg.get("warmup_teacher_temp_epochs")

    teacher_temp_warmup_iters = _resolve_warmup_iters(
        total_train_iters=total_train_iters,
        mode=teacher_cfg.get("warmup_teacher_temp_mode", "epochs"),
        warmup_epochs=t_temp_warmup_epochs,
        official_epoch_length=OFFICIAL_EPOCH_LENGTH,
        warmup_ratio=teacher_cfg.get("warmup_teacher_temp_ratio", None),
        warmup_min_steps=teacher_cfg.get("warmup_teacher_temp_min_steps", 0),
        warmup_max_steps=teacher_cfg.get("warmup_teacher_temp_max_steps", None),
    )

    
    teacher_temp = dict(
        base_value=t_temp_base,
        final_value=t_temp_base,
        total_iters=total_train_iters,
        warmup_iters=teacher_temp_warmup_iters,
        start_warmup_value=t_temp_warmup,
    )
    teacher_temp_schedule = CosineScheduler(**teacher_temp)

    
    momentum = dict(
        base_value=teacher_cfg["momentum_teacher"],
        final_value=teacher_cfg["final_momentum_teacher"],
        total_iters=total_train_iters,
    )
    momentum_schedule = CosineScheduler(**momentum)

    logger.info(
        "Scheduler setup: total_iters=%d, lr_schedule=%s, lr_warmup_iters=%d (mode=%s), teacher_temp_warmup_iters=%d (mode=%s)",
        total_train_iters,
        lr_schedule_type,
        lr_warmup_iters,
        config.optim.get("warmup_mode", "epochs"),
        teacher_temp_warmup_iters,
        teacher_cfg.get("warmup_teacher_temp_mode", "epochs"),
    )

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )



def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12, force_is_backbone=False, chunked_blocks=False):
    """
    Calculate lr decay rate for different ViT blocks.
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.
    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone") or force_is_backbone:
        if (
            ".pos_embed" in name
            or ".patch_embed" in name
            or ".mask_token" in name
            or ".cls_token" in name
            or ".reg_token" in name
            or ".register_tokens" in name
            or ".mixed_patch" in name
        ):
            layer_id = 0
        elif force_is_backbone and (
            "pos_embed" in name
            or "patch_embed" in name
            or "mask_token" in name
            or "cls_token" in name
            or "reg_token" in name
            or "register_tokens" in name
            or "mixed_patch" in name
        ):
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1
        elif chunked_blocks and "blocks." in name and "residual." not in name:
            layer_id = int(name[name.find("blocks.") :].split(".")[2]) + 1
        elif "blocks." in name and "residual." not in name:
            layer_id = int(name[name.find("blocks.") :].split(".")[1]) + 1

    return lr_decay_rate ** (num_layers + 1 - layer_id)


def _resolve_decay_model(model):
    if hasattr(model, "get_layer_id") and hasattr(model, "num_layers_for_decay"):
        return model
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "get_layer_id") and hasattr(backbone, "num_layers_for_decay"):
        return backbone
    return None


def _is_norm_param(name):
    lname = name.lower()
    return (
        "norm" in lname
        or ".bn" in lname
        or "batchnorm" in lname
        or ".ln" in lname
        or ".gn" in lname
    )

def get_params_groups_with_decay(model, lr_decay_rate=1.0, patch_embed_lr_mult=1.0):
    chunked_blocks = False
    decay_model = _resolve_decay_model(model)
    if decay_model is not None:
        n_blocks = int(getattr(decay_model, "num_layers_for_decay", 0))
    elif hasattr(model, "n_blocks"):
        logger.info("chunked fsdp")
        n_blocks = model.n_blocks
        chunked_blocks = model.chunked_blocks
    elif hasattr(model, "blocks"):
        logger.info("first code branch")
        n_blocks = len(model.blocks)
    elif hasattr(model, "backbone"):
        logger.info("second code branch")
        n_blocks = len(model.backbone.blocks)
    else:
        logger.info("else code branch")
        n_blocks = 0
    all_param_groups = []

    for name, param in model.named_parameters():
        name = name.replace("_fsdp_wrapped_module.", "")
        leaf_name = name.rsplit(".", 1)[-1]
        if not param.requires_grad:
            continue
        
        if name.startswith("probe."):
            d = {
                "params": param,
                "is_last_layer": False,
                "is_probe": True,
                "lr_multiplier": 1.0,
                "wd_multiplier": 0.0,  
                "name": name,
            }
            all_param_groups.append(d)
            continue

        if leaf_name in {"loss_b", "logit_scale", "logit_bias"}:
            d = {
                "params": param,
                "is_last_layer": False,
                "lr_multiplier": 1.0,    
                "wd_multiplier": 0.0,    
                "is_probe": False,
                "name": name,
            }
            all_param_groups.append(d)
            logger.info(f"""{name}: lr_multiplier: {d["lr_multiplier"]}, wd_multiplier: {d["wd_multiplier"]}""")
            continue

        if decay_model is not None:
            if model is decay_model or name.startswith("backbone."):
                local_name = name
                if model is not decay_model and local_name.startswith("backbone."):
                    local_name = local_name[len("backbone.") :]
                layer_id = int(decay_model.get_layer_id(local_name))
                decay_rate = lr_decay_rate ** max(n_blocks - layer_id, 0)
            else:
                decay_rate = 1.0
        else:
            decay_rate = get_vit_lr_decay_rate(
                name, lr_decay_rate, num_layers=n_blocks, force_is_backbone=n_blocks > 0, chunked_blocks=chunked_blocks
            )
        d = {"params": param, "is_last_layer": False, "lr_multiplier": decay_rate, "wd_multiplier": 1.0, "is_probe": False, "name": name}

        if "last_layer" in name:
            d.update({"is_last_layer": True})

        if (
            name.endswith(".bias")
            or _is_norm_param(name)
            or "gamma" in name
            or "layer_scale" in name
            or "pos_embed" in name
            or "cls_token" in name
        ):
            d.update({"wd_multiplier": 0.0})

        if "patch_embed" in name or "mixed_patch" in name:
            d.update({"lr_multiplier": d["lr_multiplier"] * patch_embed_lr_mult})

        all_param_groups.append(d)
        logger.info(f"""{name}: lr_multiplier: {d["lr_multiplier"]}, wd_multiplier: {d["wd_multiplier"]}""")

    return all_param_groups


def fuse_params_groups(all_params_groups, keys=("lr_multiplier", "wd_multiplier", "is_last_layer", "is_probe")):
    fused_params_groups = defaultdict(lambda: {"params": []})
    for d in all_params_groups:
        identifier = ""
        for k in keys:
            identifier += k + str(d[k]) + "_"

        for k in keys:
            fused_params_groups[identifier][k] = d[k]
        fused_params_groups[identifier]["params"].append(d["params"])

    return fused_params_groups.values()
