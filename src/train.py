import torch
import torch.nn as nn
import sys
import numpy as np
import torch.nn.functional as F

from src.utils.utils import forward_pass, perform_optimizer_step, update_teacher_ema, check_collapse, update_metric_logger, move_to_device
from src.utils.prefetch import CUDAPrefetcher

from src.utils.helpers import MetricLogger
from torch.cuda.amp import autocast
import contextlib
import logging
logger = logging.getLogger("neurojepa")


def pretrain_one_epoch(model, train_loader, optimizer, dino_scheduler, scaler, epoch, config, rank):
    model.train()
    model_ref = model.module if hasattr(model, "module") else model
    if hasattr(model_ref, "set_epoch"):
        model_ref.set_epoch(epoch)
    metric_logger = MetricLogger(delimiter="  ")
    header = f'Epoch: [{epoch}]'
    device = torch.device(f'cuda:{rank}')
    
    train_cfg = config['training']
    accum_iter = train_cfg['accum_iter']
    use_amp = train_cfg['use_amp']
    clip_grad = train_cfg.get('clip_grad', 1.0)
    max_steps = train_cfg.get('max_steps', None)
    is_dino = (config.model_chose == 'dino')
    has_teacher = hasattr(model_ref, "update_teacher")
    overlap_h2d = bool(train_cfg.get('overlap_h2d', False)) and device.type == 'cuda'

    optimizer.zero_grad()
    historical_prototypes = set()
    if config.distributed.strategy == 'ddp':
        amp_context = torch.cuda.amp.autocast()
    else:
        amp_context = contextlib.nullcontext()

    train_iterable = CUDAPrefetcher(train_loader, device) if overlap_h2d else train_loader
    reached_max_steps = False
    for data_iter_step, samples in enumerate(metric_logger.log_every(train_iterable, config['logging']['print_freq'], header)):
        if max_steps is not None and dino_scheduler.current_step >= int(max_steps):
            reached_max_steps = True
            break
        
        teacher_temp, teacher_momentum = dino_scheduler.step()
        global_step = epoch * len(train_loader) + data_iter_step
        if not overlap_h2d:
            samples = move_to_device(samples, device, non_blocking=True)

        with amp_context: 
            loss, outputs = forward_pass(
                model,
                samples,
                config,
                teacher_temp,
                update_center=is_dino,
            )
        loss_for_backward = loss / accum_iter

        if scaler is not None:
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        # update step
        if (data_iter_step + 1) % accum_iter == 0:
            perform_optimizer_step(model, optimizer, scaler, clip_grad, use_amp)
            optimizer.zero_grad()
            
            if has_teacher:
                update_teacher_ema(model, teacher_momentum)

        with torch.no_grad():
            if is_dino:
                is_logging_step = (data_iter_step > 0 and data_iter_step % 50 == 0)
                check_collapse(
                    outputs.get('t_outputs'), 
                    historical_prototypes, 
                    rank, 
                    is_logging_step=is_logging_step
                )
            
            update_metric_logger(metric_logger, loss, outputs, optimizer, teacher_temp, teacher_momentum, rank, global_step ,config)

    metric_logger.synchronize_between_processes()
    if rank == 0:
        logger.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, reached_max_steps


def finetune_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, epoch, config,
                    rank, label_scaler=None):
    """Train for one epoch"""
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    header = f'Epoch: [{epoch}]'

    train_config = config['training']
    log_config = config['logging']
    task_config = config['task']

    accum_iter = train_config['accum_iter']
    use_amp = train_config['use_amp']
    clip_grad = train_config.get('clip_grad', None)

    if train_config['freeze_encoder']:
        model.eval()
        model_ref = model.module if hasattr(model, "module") else model
        if hasattr(model_ref, "head"):
            model_ref.head.train()

    optimizer.zero_grad()
    use_wandb = bool(log_config.get('use_wandb', False))

    for data_iter_step, batch in enumerate(metric_logger.log_every(
        train_loader, log_config['print_freq'], header, enable_log=False
    )):
        global_step = epoch * len(train_loader) + data_iter_step
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            samples, labels, dataset_idx = batch
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            samples, labels = batch
            dataset_idx = None
        else:
            raise ValueError("Unexpected batch format. Expected (samples, labels) or (samples, labels, dataset_idx).")

        # Move data to GPU
        samples = samples.cuda(rank, non_blocking=True)
        labels = labels.cuda(rank, non_blocking=True)
        dataset_idx = dataset_idx.cuda(rank, non_blocking=True) if dataset_idx is not None else None

        # Forward pass with mixed precision
        with autocast(enabled=use_amp):
            model_outputs = model(samples)
            if isinstance(model_outputs, tuple):
                outputs = model_outputs[0]
            else:
                outputs = model_outputs

            if task_config['task_type'] == 'classification':
                labels = labels.squeeze().long() if labels.dim() > 1 else labels.long()

                task_loss = criterion(outputs, labels)
                _, predicted = outputs.max(1)
                correct = predicted.eq(labels).sum().item()
                accuracy = correct / labels.size(0)
            elif task_config['task_type'] == 'regression':
                target_for_loss = label_scaler.transform(labels) if label_scaler is not None else labels
                task_loss = criterion(outputs.view_as(target_for_loss), target_for_loss)
            else:
                raise ValueError(f"Unsupported task type: {task_config['task_type']}")

            loss = task_loss / accum_iter

        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()

            if (data_iter_step + 1) % accum_iter == 0:
                if clip_grad is not None:
                    scaler.unscale_(optimizer)
                    params_to_clip = list(model.parameters())
                    nn.utils.clip_grad_norm_(params_to_clip, clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
        else:
            loss.backward()

            if (data_iter_step + 1) % accum_iter == 0:
                if clip_grad is not None:
                    params_to_clip = list(model.parameters())
                    nn.utils.clip_grad_norm_(params_to_clip, clip_grad)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        # Synchronize loss across GPUs
        loss_value = loss.item() * accum_iter
        if not np.isfinite(loss_value):
            logger.error(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        metric_logger.update(loss=loss_value)
        metric_logger.update(task_loss=task_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if task_config['task_type'] == 'classification':
            metric_logger.update(acc=accuracy)

        if rank == 0 and use_wandb:
            try:
                import wandb
            except ImportError:
                wandb = None

            if wandb is not None and wandb.run is not None:
                wandb_payload = {
                    'epoch': epoch,
                    'train/step': data_iter_step,
                    'train/loss': loss_value,
                    'train/task_loss': task_loss.item(),
                    'train/lr': optimizer.param_groups[0]["lr"],
                }
                if task_config['task_type'] == 'classification':
                    wandb_payload['train/acc'] = accuracy
                wandb.log(wandb_payload, step=global_step)


    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
