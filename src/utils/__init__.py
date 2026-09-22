import torch
import numpy as np
import math
import logging
import torch.distributed as dist
from .optim import build_schedulers, apply_optim_scheduler
logger = logging.getLogger("neurojepa")

def apply_scaling_rules_to_cfg(cfg):  

    if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
    else:
        world_size = 1
    base_lr = cfg.optim.base_lr
    cfg.optim.lr = base_lr
    if cfg.optim.scaling_rule == "sqrt_wrt_1024":
        cfg.optim.lr *= math.sqrt(cfg.data.batch_size * world_size * cfg.training.accum_iter / 1024.0)
        logger.info(f"sqrt scaling learning rate; base: {base_lr}, new: {cfg.optim.lr}")
    elif cfg.optim.scaling_rule == "linear":
        scale_factor = cfg.data.batch_size * world_size * cfg.training.accum_iter / 256.0
        cfg.optim.lr *= scale_factor
        logger.info(f"Linear scaling LR (ref 256); base: {base_lr}, batch: {cfg.data.batch_size * world_size * cfg.training.accum_iter}, new: {cfg.optim.lr}")
    elif cfg.optim.scaling_rule == "none":
        cfg.optim.lr = base_lr
        logger.info(f"No scaling LR; using base: {base_lr}")
    else:
        raise NotImplementedError
    return cfg

def set_seed(seed, rank=0):
    """Set random seed for reproducibility"""
    seed = seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))

class DINOSchedulerWrapper:
    def __init__(self, config, optimizer, start_epoch=0):
        self.optimizer = optimizer
        self.config = config
        
        (
            self.lr_schedule,
            self.wd_schedule,
            self.momentum_schedule,
            self.teacher_temp_schedule,
            self.last_layer_lr_schedule,
        ) = build_schedulers(config)
        
        self.current_step = start_epoch * config['training']['OFFICIAL_EPOCH_LENGTH']
        self.probe_lr = self.config.optim.get("probe_lr", None)
        self.probe_wd = self.config.optim.get("probe_wd", None)
        self.tail_resume_state = None

    def _scheduled_value(self, schedule, step=None):
        if len(schedule) == 0:
            return None
        if step is None:
            step = max(int(self.current_step) - 1, 0)
        step = max(0, min(int(step), len(schedule) - 1))
        return float(schedule[step])

    def _optimizer_base_value(self, field, multiplier_key, fallback, skip_last_layer=True, require_last_layer=False):
        for group in self.optimizer.param_groups:
            if group.get("is_probe", False):
                continue
            if skip_last_layer and group.get("is_last_layer", False):
                continue
            if require_last_layer and not group.get("is_last_layer", False):
                continue
            multiplier = float(group.get(multiplier_key, 1.0))
            if multiplier == 0:
                continue
            value = group.get(field, None)
            if value is None:
                continue
            return float(value) / multiplier
        if fallback is None:
            raise ValueError(f"Could not infer optimizer field {field}")
        return float(fallback)

    @staticmethod
    def _cosine_tail(start_value, final_value, steps):
        steps = int(steps)
        if steps <= 0:
            return np.array([], dtype=np.float64)
        if steps == 1:
            return np.array([float(final_value)], dtype=np.float64)
        iters = np.arange(steps, dtype=np.float64)
        return float(final_value) + 0.5 * (float(start_value) - float(final_value)) * (
            1.0 + np.cos(np.pi * iters / float(steps - 1))
        )

    def _replace_schedule_tail(self, schedule, start_step, start_value, final_value, total_steps):
        start_step = int(start_step)
        total_steps = int(total_steps)
        if total_steps <= start_step:
            raise ValueError(f"total_steps ({total_steps}) must be > start_step ({start_step})")

        prefix_len = min(len(schedule), start_step)
        prefix = np.asarray([schedule[i] for i in range(prefix_len)], dtype=np.float64)
        if prefix_len < start_step:
            pad = np.full(start_step - prefix_len, float(start_value), dtype=np.float64)
            prefix = np.concatenate([prefix, pad])

        tail = self._cosine_tail(start_value, final_value, total_steps - start_step)
        return np.concatenate([prefix, tail])

    def _apply_tail_resume_state(self, state):
        start_step = int(state["start_step"])
        total_steps = int(state["total_steps"])
        self.lr_schedule = self._replace_schedule_tail(
            self.lr_schedule, start_step, state["start_lr"], state["final_lr"], total_steps
        )
        self.last_layer_lr_schedule = self._replace_schedule_tail(
            self.last_layer_lr_schedule,
            start_step,
            state.get("start_last_layer_lr", state["start_lr"]),
            state.get("final_last_layer_lr", state["final_lr"]),
            total_steps,
        )
        self.wd_schedule = self._replace_schedule_tail(
            self.wd_schedule, start_step, state["start_wd"], state["final_wd"], total_steps
        )
        self.momentum_schedule = self._replace_schedule_tail(
            self.momentum_schedule, start_step, state["start_momentum"], state["final_momentum"], total_steps
        )
        self.teacher_temp_schedule = self._replace_schedule_tail(
            self.teacher_temp_schedule,
            start_step,
            state["start_teacher_temp"],
            state["final_teacher_temp"],
            total_steps,
        )
        self.tail_resume_state = dict(state)

    def rebuild_tail_from_current_lr(self, extra_steps, final_lr=None, final_wd=None):
        """
        Rebuild schedules so resumed training continues smoothly from the
        checkpoint LR instead of jumping onto a newly stretched cosine curve.
        """
        extra_steps = int(extra_steps)
        if extra_steps <= 0:
            raise ValueError("extra_steps must be positive")

        start_step = int(self.current_step)
        total_steps = start_step + extra_steps
        final_lr = float(self.config.optim.get("min_lr") if final_lr is None else final_lr)
        final_wd = float(self.config.optim.get("weight_decay_end") if final_wd is None else final_wd)

        start_lr = self._optimizer_base_value(
            "lr",
            "lr_multiplier",
            fallback=self._scheduled_value(self.lr_schedule),
            skip_last_layer=True,
        )
        start_wd = self._optimizer_base_value(
            "weight_decay",
            "wd_multiplier",
            fallback=self._scheduled_value(self.wd_schedule),
            skip_last_layer=True,
        )
        start_last_layer_lr = self._optimizer_base_value(
            "lr",
            "lr_multiplier",
            fallback=self._scheduled_value(self.last_layer_lr_schedule),
            skip_last_layer=False,
            require_last_layer=True,
        )
        teacher_cfg = self.config.get("teacher", {})
        start_momentum = self._scheduled_value(self.momentum_schedule)
        final_momentum = teacher_cfg.get("final_momentum_teacher", start_momentum)
        start_teacher_temp = self._scheduled_value(self.teacher_temp_schedule)
        final_teacher_temp = teacher_cfg.get("teacher_temp", start_teacher_temp)

        state = {
            "start_step": start_step,
            "total_steps": total_steps,
            "extra_steps": extra_steps,
            "start_lr": float(start_lr),
            "final_lr": float(final_lr),
            "start_last_layer_lr": float(start_last_layer_lr),
            "final_last_layer_lr": float(final_lr),
            "start_wd": float(start_wd),
            "final_wd": float(final_wd),
            "start_momentum": float(start_momentum),
            "final_momentum": float(final_momentum),
            "start_teacher_temp": float(start_teacher_temp),
            "final_teacher_temp": float(final_teacher_temp),
        }
        self._apply_tail_resume_state(state)
        logger.info(
            "Rebuilt resume tail cosine: start_step=%d total_steps=%d start_lr=%.8g final_lr=%.8g start_wd=%.8g final_wd=%.8g",
            start_step,
            total_steps,
            start_lr,
            final_lr,
            start_wd,
            final_wd,
        )
        return state

    def step(self):
        """
        update LR and WD
        """
        if self.current_step >= len(self.lr_schedule):
            self.current_step = len(self.lr_schedule) - 1
            
        lr = self.lr_schedule[self.current_step]
        wd = self.wd_schedule[self.current_step]
        mom = self.momentum_schedule[self.current_step]
        teacher_temp = self.teacher_temp_schedule[self.current_step]
        last_layer_lr = self.last_layer_lr_schedule[self.current_step]

        apply_optim_scheduler(self.optimizer, lr, wd, last_layer_lr, self.probe_lr, self.probe_wd)

        self.current_step += 1
        
        return teacher_temp, mom

    def state_dict(self):
        state = {'current_step': self.current_step}
        if self.tail_resume_state is not None:
            state['tail_resume_state'] = self.tail_resume_state
        return state

    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']
        tail_state = state_dict.get('tail_resume_state')
        if tail_state is not None:
            self._apply_tail_resume_state(tail_state)


def create_downstream_optimizer(model, config):
    """
    Create optimizer with Layer-wise Learning Rate Decay (LLRD) support.
    """
    train_config = config['optim']
    
    base_lr = train_config['learning_rate']
    weight_decay = train_config['weight_decay']
    head_lr = train_config.get('head_lr', base_lr)
    layer_decay = train_config.get('layer_decay', 1.0)
    head_keys = ('head', 'prototypes', 'prototype_logit_scale')

    def _use_no_decay(name, param):
        return (
            param.ndim <= 1
            or name.endswith(".bias")
            or "pos_embed" in name
            or "cls_token" in name
            or "reg_token" in name
            or "register_tokens" in name
            or "mask_token" in name
        )

    if layer_decay >= 1.0:
        print(f"Layer decay not enabled (value: {layer_decay}). Using standard optimizer.")
        encoder_decay_params = []
        encoder_no_decay_params = []
        head_decay_params = []
        head_no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            is_head = any(k in name for k in head_keys)
            use_no_decay = _use_no_decay(name, param)

            if is_head and use_no_decay:
                head_no_decay_params.append(param)
            elif is_head:
                head_decay_params.append(param)
            elif use_no_decay:
                encoder_no_decay_params.append(param)
            else:
                encoder_decay_params.append(param)

        param_groups = []
        if encoder_decay_params:
            param_groups.append({'params': encoder_decay_params, 'lr': base_lr, 'weight_decay': weight_decay})
        if encoder_no_decay_params:
            param_groups.append({'params': encoder_no_decay_params, 'lr': base_lr, 'weight_decay': 0.0})
        if head_decay_params:
            param_groups.append({'params': head_decay_params, 'lr': head_lr, 'weight_decay': weight_decay})
        if head_no_decay_params:
            param_groups.append({'params': head_no_decay_params, 'lr': head_lr, 'weight_decay': 0.0})
    else:
        print(f"Applying Layer-wise Learning Rate Decay (decay rate: {layer_decay})")
        param_groups = []
        try:
            num_layers = len(model.blocks) 
        except:
            num_layers = config['model'].get('depth', 12)
            
        print(f"Detected total layers: {num_layers}")
        total_depth = num_layers + 1

        def get_layer_id(name):
            if "patch_embed" in name or "pos_embed" in name or "cls_token" in name or "mixed_patch" in name:
                return 0
            elif name.startswith("blocks") or name.startswith("layers"):
                try:
                    return int(name.split('.')[1]) + 1
                except:
                    return 0
            
            elif name.startswith("norm.") or name.startswith("fc_norm."):
                return total_depth
            
            else:
                return 0

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if _use_no_decay(name, param):
                this_weight_decay = 0.0
            else:
                this_weight_decay = weight_decay

            if any(k in name for k in head_keys):
                param_groups.append({
                    'params': [param],
                    'lr': head_lr, 
                    'weight_decay': this_weight_decay,
                    'name': name
                })
                continue
            layer_id = get_layer_id(name)
            
            scale = layer_decay ** (total_depth - layer_id)
            group_lr = base_lr * scale
            
            param_groups.append({
                'params': [param],
                'lr': group_lr,
                'weight_decay': this_weight_decay,
                'name': name
            })

    if train_config['optimizer'].lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=tuple(train_config['betas']),
            weight_decay=weight_decay 
        )
    elif train_config['optimizer'].lower() == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            momentum=train_config.get('momentum', 0.9),
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {train_config['optimizer']}")

    return optimizer


def create_downstream_scheduler(optimizer, config, steps_per_epoch):
    """Create learning rate scheduler"""
    train_config = config['optim']
    total_steps = train_config['epochs'] * steps_per_epoch
    warmup_steps = train_config['warmup_epochs'] * steps_per_epoch

    if train_config['lr_scheduler'].lower() == 'cosine':
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine annealing
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(train_config['min_lr'] / train_config['learning_rate'],
                          0.5 * (1.0 + np.cos(np.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"Unsupported scheduler: {train_config['lr_scheduler']}")

    return scheduler
