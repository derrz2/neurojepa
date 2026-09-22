import contextlib
import logging
from typing import List, Tuple, Union

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

logger = logging.getLogger("neurojepa")


def forward_pass(model, samples, cfg, teacher_temp=None, update_center=False):
    if cfg.model_chose == 'dino':
        l, lg, ll, lk, prob_l, tout = model(samples, teacher_temp=teacher_temp, update_center=update_center)
        return l, {'loss_global': lg, 'loss_local': ll, 'loss_koleo': lk, 'probe_loss': prob_l, 't_outputs': tout}
    elif cfg.model_chose == 'byol':
        l, ssl_l, prob_l, ali_metric, coll_metrics = model(samples)
        return l, {
            'byol_loss': ssl_l,
            'probe_loss': prob_l,
            'ali_metric': ali_metric,
            'collapse_metrics': coll_metrics,
        }
    elif cfg.model_chose == 'vicreg':
        l, inv_l, var_l, cov_l, prob_l, ali_metric, coll_metrics = model(samples)
        return l, {
            'inv_loss': inv_l,
            'var_loss': var_l,
            'cov_loss': cov_l,
            'probe_loss': prob_l,
            'ali_metric': ali_metric,
            'collapse_metrics': coll_metrics,
        }
    elif cfg.model_chose == 'simclr':
        l, ssl_l, prob_l, ali_metric, coll_metrics = model(samples)
        return l, {
            'simclr_loss': ssl_l,
            'probe_loss': prob_l,
            'ali_metric': ali_metric,
            'collapse_metrics': coll_metrics,
        }
    elif cfg.model_chose == 'simsiam':
        l, ssl_l, prob_l, ali_metric, coll_metrics = model(samples)
        return l, {
            'simsiam_loss': ssl_l,
            'probe_loss': prob_l,
            'ali_metric': ali_metric,
            'collapse_metrics': coll_metrics,
        }
    elif cfg.model_chose == 'swav':
        l, ssl_l, prob_l, ali_metric, coll_metrics = model(samples)
        return l, {
            'swav_loss': ssl_l,
            'probe_loss': prob_l,
            'ali_metric': ali_metric,
            'collapse_metrics': coll_metrics,
        }
    elif cfg.model_chose == 'ijepa':
        l, ssl_l, prob_l, ali_metric, coll_metrics = model(samples)
        return l, {
            'ijepa_loss': ssl_l,
            'probe_loss': prob_l,
            'ali_metric': ali_metric,
            'collapse_metrics': coll_metrics,
        }
    elif cfg.model_chose == 'mae':
        l, recon_l, mask_ratio = model(samples)
        return l, {
            'recon_loss': recon_l,
            'mask_ratio': mask_ratio,
        }
    elif cfg.model_chose == 'simmim':
        l, recon_l, mask_ratio = model(samples)
        return l, {
            'recon_loss': recon_l,
            'mask_ratio': mask_ratio,
        }
    elif cfg.model_chose == 'lejepa':
        l, inv_l, sigreg_l, prob_l, extra_metrics, aliment_metric, coll_metrics = model(samples)
        outputs = {
            'inv_loss': inv_l,
            'probe_loss': prob_l,
            'ali_metric': aliment_metric,
            'collapse_metrics': coll_metrics,
        }
        if bool(getattr(cfg.model.lejepa, "sigreg_enable", True)):
            outputs['sigreg_loss'] = sigreg_l
        outputs.update(extra_metrics)
        return l, outputs
    elif cfg.model_chose == 'lejepa_sliced_prior':
        l, inv_l, prior_l, sigreg_l, prob_l, extra_metrics, aliment_metric, coll_metrics = model(samples)
        outputs = {
            'inv_loss': inv_l,
            'prior_loss': prior_l,
            'sigreg_loss': sigreg_l,
            'probe_loss': prob_l,
            'ali_metric': aliment_metric,
            'collapse_metrics': coll_metrics,
        }
        outputs.update(extra_metrics)
        return l, outputs
    elif cfg.model_chose == 'leworld':
        l, pred_l, sigreg_l, prob_l, ali_metric, coll_metrics, pooled_pred_l, token_pred_l = model(samples)
        outputs = {
            'pred_loss': pred_l,
            'pooled_pred_loss': pooled_pred_l,
            'token_pred_loss': token_pred_l,
            'probe_loss': prob_l,
            'ali_metric': ali_metric,
            'collapse_metrics': coll_metrics,
        }
        if bool(getattr(cfg.model.leworld, "sigreg_enable", True)):
            outputs['sigreg_loss'] = sigreg_l
        return l, outputs
    elif cfg.model_chose == 'lpjepa':
        l, inv_l, reg_l, prob_l, coll_metrics = model(samples)
        return l, {
            'inv_loss': inv_l,
            'sigreg_loss': reg_l,
            'probe_loss': prob_l,
            'collapse_metrics': coll_metrics,
        }
    raise ValueError(f"Unsupported model_chose: {cfg.model_chose}")


def perform_optimizer_step(model, optimizer, scaler, clip_grad, use_amp):
    if use_amp and scaler is not None:
        if clip_grad is not None:
            scaler.unscale_(optimizer)
            if hasattr(model, 'clip_grad_norm_'):
                model.clip_grad_norm_(clip_grad)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()
    else:
        if clip_grad is not None:
            if hasattr(model, 'clip_grad_norm_'):
                model.clip_grad_norm_(clip_grad)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()


def update_teacher_ema(model, momentum):
    is_fsdp = isinstance(model, FSDP)
    ctx = model.summon_full_params(model, writeback=True) if is_fsdp else contextlib.nullcontext()
    with ctx:
        m = model.module if hasattr(model, 'module') else model
        if hasattr(m, 'update_teacher'):
            m.update_teacher(momentum)


def check_collapse(t_outputs, historical_set, rank, is_logging_step=False):
    if t_outputs is None:
        return

    local_indices = t_outputs.argmax(dim=-1)
    gathered_indices = concat_all_gather(local_indices)
    if rank == 0:
        historical_set.update(gathered_indices.cpu().numpy().tolist())
        if is_logging_step:
            logger.info(f"Unique prototypes (last 50 steps): {len(historical_set)}")
            historical_set.clear()


def update_metric_logger(metric_logger, loss, outputs, optimizer, t_temp, t_momentum, rank, step, cfg):
    loss_val = loss.item() if torch.is_tensor(loss) else loss
    lr = optimizer.param_groups[0]["lr"]
    wd_group = next(
        (param_group for param_group in optimizer.param_groups if param_group.get("wd_multiplier", 1.0) > 0),
        optimizer.param_groups[0],
    )
    wd = wd_group["weight_decay"]

    metric_logger.update(loss=loss_val)
    metric_logger.update(lr=lr)
    metric_logger.update(wd=wd)

    wandb_dict = {
        "train/loss": loss_val,
        "train/lr": lr,
        "train/wd": wd,
    }

    if cfg.model_chose == 'dino':
        for k in ['loss_global', 'loss_local', 'loss_koleo']:
            if k in outputs:
                metric_logger.update(**{k: outputs[k]})
        metric_logger.update(teacher_temp=t_temp)
        metric_logger.update(teacher_momentum=t_momentum)
    elif cfg.model_chose == 'byol':
        for k in ['byol_loss', 'ali_metric']:
            if k in outputs:
                val = outputs[k].item() if torch.is_tensor(outputs[k]) else outputs[k]
                metric_logger.update(**{k: val})
                wandb_dict[f"train/{k}"] = val
        metric_logger.update(teacher_momentum=t_momentum)
        wandb_dict["train/teacher_momentum"] = t_momentum
    elif cfg.model_chose == 'vicreg':
        for k in ['inv_loss', 'var_loss', 'cov_loss', 'ali_metric']:
            if k in outputs:
                val = outputs[k].item() if torch.is_tensor(outputs[k]) else outputs[k]
                metric_logger.update(**{k: val})
                wandb_dict[f"train/{k}"] = val
    elif cfg.model_chose == 'simclr':
        for k in ['simclr_loss', 'ali_metric']:
            if k in outputs:
                val = outputs[k].item() if torch.is_tensor(outputs[k]) else outputs[k]
                metric_logger.update(**{k: val})
                wandb_dict[f"train/{k}"] = val
    elif cfg.model_chose == 'simsiam':
        for k in ['simsiam_loss', 'ali_metric']:
            if k in outputs:
                val = outputs[k].item() if torch.is_tensor(outputs[k]) else outputs[k]
                metric_logger.update(**{k: val})
                wandb_dict[f"train/{k}"] = val
    elif cfg.model_chose == 'swav':
        for k in ['swav_loss', 'ali_metric']:
            if k in outputs:
                val = outputs[k].item() if torch.is_tensor(outputs[k]) else outputs[k]
                metric_logger.update(**{k: val})
                wandb_dict[f"train/{k}"] = val
    elif cfg.model_chose == 'ijepa':
        for k in ['ijepa_loss', 'ali_metric']:
            if k in outputs:
                val = outputs[k].item() if torch.is_tensor(outputs[k]) else outputs[k]
                metric_logger.update(**{k: val})
                wandb_dict[f"train/{k}"] = val
        metric_logger.update(teacher_momentum=t_momentum)
        wandb_dict["train/teacher_momentum"] = t_momentum
    elif cfg.model_chose in {'mae', 'simmim'}:
        for k in ['recon_loss', 'mask_ratio']:
            if k in outputs:
                val = outputs[k].item() if torch.is_tensor(outputs[k]) else outputs[k]
                metric_logger.update(**{k: val})
                wandb_dict[f"train/{k}"] = val
    elif cfg.model_chose == 'lejepa':
        metric_keys = ['inv_loss', 'ali_metric', 'anchor_loss', 'all_loss', 'weighted_inv_loss', 'weighted_sigreg_loss']
        if 'local_loss' in outputs:
            metric_keys.append('local_loss')
        if bool(getattr(cfg.model.lejepa, "sigreg_enable", True)):
            metric_keys.append('sigreg_loss')
        for k in metric_keys:
            if k in outputs:
                val = outputs[k].item() if torch.is_tensor(outputs[k]) else outputs[k]
                metric_logger.update(**{k: val})
                wandb_dict[f"train/{k}"] = val
    elif cfg.model_chose == 'lejepa_sliced_prior':
        for k, v in outputs.items():
            if k in {'probe_loss', 'collapse_metrics'}:
                continue
            val = v.item() if torch.is_tensor(v) else v
            metric_logger.update(**{k: val})
            wandb_dict[f"train/{k}"] = val
    elif cfg.model_chose == 'leworld':
        metric_keys = ['pred_loss', 'pooled_pred_loss', 'token_pred_loss', 'ali_metric']
        if bool(getattr(cfg.model.leworld, "sigreg_enable", True)):
            metric_keys.append('sigreg_loss')
        for k in metric_keys:
            if k in outputs:
                val = outputs[k].item() if torch.is_tensor(outputs[k]) else outputs[k]
                metric_logger.update(**{k: val})
                wandb_dict[f"train/{k}"] = val
    elif cfg.model_chose == 'lpjepa':
        inv_weight = float(cfg.model.lpjepa.invariance_loss_weight)
        reg_weight = float(cfg.model.lpjepa.rdm_reg_loss_weight)
        metric_weights = {
            'inv_loss': inv_weight,
            'sigreg_loss': reg_weight,
        }
        for k in ['inv_loss', 'sigreg_loss']:
            if k in outputs:
                val = outputs[k].item() if torch.is_tensor(outputs[k]) else outputs[k]
                weighted_val = val * metric_weights[k]
                metric_logger.update(**{k: weighted_val})
                wandb_dict[f"train/{k}"] = weighted_val

    if 'probe_loss' in outputs and cfg.training.probe_val:
        probe_val = outputs['probe_loss'].item() if torch.is_tensor(outputs['probe_loss']) else outputs['probe_loss']
        metric_logger.update(probe_loss=probe_val)
        wandb_dict["train/probe_loss"] = probe_val
    if 'collapse_metrics' in outputs:
        for k, v in outputs['collapse_metrics'].items():
            val = v.item() if torch.is_tensor(v) else v
            metric_logger.update(**{k: val})
            wandb_dict[f"collapse/{k}"] = val

    if rank == 0 and cfg.logging.use_wandb:
        import wandb

        wandb.log(wandb_dict, step=step)


def _to_2tuple(x: Union[int, Tuple[int, int], List[int]]) -> Tuple[int, int]:
    if isinstance(x, int):
        return (x, x)
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return int(x[0]), int(x[1])
    raise ValueError(f"Expected int or 2-tuple/list, got {x}")


def to_3tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x, x)


@torch.no_grad()
def concat_all_gather(tensor):
    world_size = dist.get_world_size()
    tensors_gather = [torch.ones_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=0)


class LabelScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, labels):
        return (labels - self.mean) / self.std


def move_to_device(batch, device, non_blocking=True):
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, list):
        return [move_to_device(item, device, non_blocking=non_blocking) for item in batch]
    if isinstance(batch, tuple):
        return tuple(move_to_device(item, device, non_blocking=non_blocking) for item in batch)
    if isinstance(batch, dict):
        return {key: move_to_device(value, device, non_blocking=non_blocking) for key, value in batch.items()}
    return batch


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


class GatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized():
            output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(output, x)
        else:
            output = [x]
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        if dist.is_available() and dist.is_initialized():
            all_gradients = torch.stack(grads)
            dist.all_reduce(all_gradients)
            grad_out = all_gradients[get_rank()]
        else:
            grad_out = grads[0]
        return grad_out


def gather(X, dim=0):
    return torch.cat(GatherLayer.apply(X), dim=dim)
