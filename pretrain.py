import os
import sys
import argparse
import time
import datetime
import math
from pathlib import Path

import logging
logger = logging.getLogger("neurojepa")

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.cuda.amp import GradScaler
from torch.distributed.elastic.multiprocessing.errors import record

from src.models import build_pretrain_model

from subject_session_data import create_pretrain_dataloaders
from src.data import create_pretrain_hard_val_dataloader

from src.utils.logging_utils import load_config, save_config, setup_logger
from src.distributed.dist_ddp import setup_distributed, cleanup_distributed
from src.distributed.fsdp_helper import wrap_pretrain_modules_fsdp
from src.utils.checkpoint import get_unified_state_dict, save_checkpoint, load_checkpoint
from src.utils import set_seed, build_optimizer, DINOSchedulerWrapper, apply_scaling_rules_to_cfg
from src.train import pretrain_one_epoch
from src.eval import validate, validate_probe_acc, validate_rankme
import warnings
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")


def _collect_model_metadata(model, config, world_size):
    backbone = getattr(model, "backbone", None)
    proj = getattr(model, "proj", None)
    augment_cfg = getattr(config.data, "augment", None)
    train_view_mode = getattr(augment_cfg, "view_mode", "none")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metadata = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "global_batch_size": int(config.data.batch_size) * int(world_size) * int(config.training.accum_iter),
        "train_view_mode": str(train_view_mode),
    }
    if backbone is not None:
        metadata["backbone_params"] = sum(p.numel() for p in backbone.parameters())
        metadata["tokens_per_view"] = int(getattr(backbone, "num_patches", 0)) + int(getattr(backbone, "num_prefix_tokens", 0))
        metadata["embed_dim"] = int(getattr(backbone, "embed_dim", 0))
        metadata["depth"] = int(getattr(backbone, "depth", 0))
    if proj is not None:
        metadata["projector_params"] = sum(p.numel() for p in proj.parameters())
    return metadata

@record
def main():
    """Main training function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='fMRI Pretraining')
    parser.add_argument('--config', type=str, default='configs/pretrain_config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument(
        '--cosine_tail_extra_steps',
        type=int,
        default=None,
        help='When resuming, rebuild scheduler tails from the restored current LR for this many extra optimizer steps.',
    )
    parser.add_argument(
        '--cosine_tail_min_lr',
        type=float,
        default=None,
        help='Final LR for --cosine_tail_extra_steps. Defaults to optim.cosine_tail_min_lr or optim.min_lr.',
    )
    parser.add_argument(
        '--cosine_tail_final_wd',
        type=float,
        default=None,
        help='Final weight decay for --cosine_tail_extra_steps. Defaults to optim.cosine_tail_final_wd or optim.weight_decay_end.',
    )
    parser.add_argument('--no_val', action='store_true', help='Disable validation')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (overrides config)')
    parser.add_argument(
        '--activation_checkpoint',
        action='store_true',
        help='Enable activation checkpointing for pretraining (overrides config.training.activation_checkpointing).',
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    config['experiment']['resume'] = args.resume or config['experiment'].get('resume')
    config['experiment']['output_dir'] = args.output_dir or config['experiment'].get('output_dir')
    if args.cosine_tail_extra_steps is not None:
        config.setdefault('optim', {})
        config['optim']['cosine_tail_extra_steps'] = int(args.cosine_tail_extra_steps)
    if args.cosine_tail_min_lr is not None:
        config.setdefault('optim', {})
        config['optim']['cosine_tail_min_lr'] = float(args.cosine_tail_min_lr)
    if args.cosine_tail_final_wd is not None:
        config.setdefault('optim', {})
        config['optim']['cosine_tail_final_wd'] = float(args.cosine_tail_final_wd)
    if args.activation_checkpoint:
        config.setdefault('training', {})
        config['training']['activation_checkpointing'] = True

    is_distributed, rank, world_size, gpu = setup_distributed()
    if rank == 0 and config.logging.use_wandb:
        import wandb  
        wandb_name = config.logging.get("wandb_name") or config.experiment.get("name") or f"run-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
        wandb_group = config.logging.get("wandb_group", None)
        wandb_job_type = config.logging.get("wandb_job_type", None)
        wandb_tags = config.logging.get("wandb_tags", None)
        wandb.init(
            project=config.logging.wandb_project, 
            name=wandb_name,
            group=wandb_group,
            job_type=wandb_job_type,
            tags=wandb_tags,
            config=config, 
            resume="allow" if args.resume else None
        )
        wandb.define_metric("epoch")
        wandb.define_metric("val/*", step_metric="epoch")
    apply_scaling_rules_to_cfg(config)

    set_seed(config['experiment']['seed'], rank)
    train_strategy = config.distributed.strategy

    # Create output directories
    output_dir = Path(config['experiment']['output_dir'])
    checkpoint_dir = output_dir / 'checkpoints'

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        save_config(config, output_dir / 'config.yaml')

        # Setup text log file
        with open(output_dir / 'training_log.txt', 'w') as f:
            f.write(f"Training started at {datetime.datetime.now()}\nConfig: {args.config}\n")

    logger = setup_logger(output_dir, name="neurojepa", rank=rank)
    if is_distributed: dist.barrier()

    # Print configuration
    if rank == 0:
        logger.info(f"Distributed: {is_distributed} (Strategy: {train_strategy}, World Size: {world_size})")
        activation_checkpoint_cfg = config.training.get("activation_checkpointing", False)
        if bool(activation_checkpoint_cfg):
            logger.info(
                "Activation checkpointing is enabled for ViT transformer blocks during pretraining: %s",
                activation_checkpoint_cfg,
            )

    if rank == 0:
        logger.info(f"Creating model with strategy: {train_strategy}")

    model = build_pretrain_model(config)
    torch.cuda.set_device(gpu)
    model = model.cuda(gpu)

    if config['training']['freeze_encoder']:
        if rank == 0:
            logger.info("Freezing encoder weights. Only the head will be trained.")
        for _, param in model.backbone.named_parameters():
                param.requires_grad = False

    if is_distributed:
        if train_strategy == 'ddp':
            model = DDP(model, device_ids=[gpu], find_unused_parameters=True)
        elif train_strategy == 'fsdp':
            wrapped_modules = wrap_pretrain_modules_fsdp(model)
            if rank == 0:
                logger.info(f"FSDP wrapped modules: {', '.join(wrapped_modules) if wrapped_modules else '<none>'}")

    model_without_ddp = model.module if hasattr(model, 'module') else model
    model_metadata = _collect_model_metadata(model_without_ddp, config, world_size)

    train_loader, val_loader, train_sampler = create_pretrain_dataloaders(
        config, is_distributed, rank, world_size
    )
    hard_val_loader = create_pretrain_hard_val_dataloader(
        config, is_distributed, rank, world_size
    )
    val_fn = validate 

    if rank == 0:
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        if hard_val_loader is not None:
            logger.info(f"Hard validation samples: {len(hard_val_loader.dataset)}")
        logger.info(f"Batches per epoch: {len(train_loader)}")
        logger.info("Model metadata: " + " | ".join([f"{k}: {v}" for k, v in model_metadata.items()]))
        if config.logging.use_wandb:
            import wandb
            wandb.config.update(model_metadata, allow_val_change=True)

    actual_epoch_length = max(len(train_loader), 1)
    if config.training.OFFICIAL_EPOCH_LENGTH != actual_epoch_length:
        if rank == 0:
            logger.info(
                f"Overriding OFFICIAL_EPOCH_LENGTH from {config.training.OFFICIAL_EPOCH_LENGTH} "
                f"to actual train_loader length {actual_epoch_length}"
            )
        config.training.OFFICIAL_EPOCH_LENGTH = actual_epoch_length


    # Create optimizer and scheduler
    optimizer = build_optimizer(config, model_without_ddp.get_params_groups())
    dino_scheduler = DINOSchedulerWrapper(config, optimizer, start_epoch=0)
    # bf16 FSDP does not need loss scaling; GradScaler is only for fp16-style AMP.
    scaler = None if train_strategy == 'fsdp' else GradScaler()

    # Load checkpoint if resuming
    start_epoch, best_metric = 0, float('inf')
    resume_path = config['experiment'].get('resume')
    if resume_path:
        start_epoch, _, best_metric = load_checkpoint(
            checkpoint_path=resume_path,
            model=model, 
            optimizer=optimizer,
            scheduler=dino_scheduler,
            scaler=scaler,
            train_strategy=train_strategy
        )
        tail_extra_steps = config.optim.get("cosine_tail_extra_steps", None)
        if tail_extra_steps is not None:
            tail_extra_steps = int(tail_extra_steps)
            if tail_extra_steps <= 0:
                raise ValueError("cosine_tail_extra_steps must be positive when provided")

            final_lr = config.optim.get("cosine_tail_min_lr", config.optim.get("min_lr", None))
            final_wd = config.optim.get("cosine_tail_final_wd", config.optim.get("weight_decay_end", None))
            tail_state = dino_scheduler.rebuild_tail_from_current_lr(
                extra_steps=tail_extra_steps,
                final_lr=final_lr,
                final_wd=final_wd,
            )

            config['training']['max_steps'] = int(tail_state["total_steps"])
            official_epoch_length = max(int(config['training']['OFFICIAL_EPOCH_LENGTH']), 1)
            tail_epochs = int(math.ceil(tail_extra_steps / official_epoch_length))
            config['optim']['epochs'] = max(
                int(config['optim']['epochs']),
                int(start_epoch) + tail_epochs + 1,
            )
            if rank == 0:
                logger.info(
                    "Cosine tail resume enabled: current_step=%d, extra_steps=%d, new max_steps=%d, epochs=%d",
                    int(tail_state["start_step"]),
                    tail_extra_steps,
                    int(config['training']['max_steps']),
                    int(config['optim']['epochs']),
                )
                save_config(config, output_dir / 'config.yaml')
    elif config.optim.get("cosine_tail_extra_steps", None) is not None:
        raise ValueError("cosine_tail_extra_steps requires --resume or experiment.resume")

    # Training loop
    if rank == 0:
        logger.info(f"Training from epoch {start_epoch} to {config['optim']['epochs']}")

    start_time = time.time()

    for epoch in range(start_epoch, config['optim']['epochs']):
        train_dataset = getattr(train_loader, "dataset", None)
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Train for one epoch
        train_stats, reached_max_steps = pretrain_one_epoch(
            model, train_loader, optimizer, dino_scheduler, scaler,
            epoch, config, rank
        )
        if rank == 0:
            logger.info(f"Epoch {epoch} Training - " + " | ".join([f"{k}: {v:.4f}" for k, v in train_stats.items()]))

        is_last_epoch = epoch == config['optim']['epochs'] - 1
        is_final_train_epoch = is_last_epoch or reached_max_steps
        is_val_epoch = not args.no_val and (
            epoch % config['validation']['val_freq'] == 0 or is_final_train_epoch
        )
        is_best = False

        # Validate
        val_stats = {}
        hard_val_stats = {}
        if is_val_epoch:
            val_temp = None
            if len(dino_scheduler.teacher_temp_schedule) > 0:
                step_idx = min(max(dino_scheduler.current_step - 1, 0), len(dino_scheduler.teacher_temp_schedule) - 1)
                val_temp = float(dino_scheduler.teacher_temp_schedule[step_idx])
            val_stats = val_fn(model, val_loader, epoch, rank, config, teacher_temp=val_temp)
            if hard_val_loader is not None:
                hard_val_stats = validate(model, hard_val_loader, epoch, rank, config, teacher_temp=val_temp)
            rankme_loader = hard_val_loader if hard_val_loader is not None else val_loader
            rankme_stats = validate_rankme(model, rankme_loader, epoch, rank, config)
            if rankme_stats:
                val_stats.update(rankme_stats)
            selection_source = hard_val_stats if hard_val_stats else val_stats
            selection_metric = config['validation'].get('selection_metric', 'loss')
            selected_value = selection_source.get(selection_metric, selection_source.get('loss', float('inf')))
            
            if rank == 0:
                logger.info(f"Epoch {epoch} Val - " + " | ".join([f"{k}: {v:.4f}" for k, v in val_stats.items()]))
                if hard_val_stats:
                    logger.info(f"Epoch {epoch} Hard Val - " + " | ".join([f"{k}: {v:.4f}" for k, v in hard_val_stats.items()]))
                save_best_enabled = bool(config['validation'].get('save_best', True))
                if selected_value < best_metric:
                    best_metric = selected_value
                    is_best = save_best_enabled
                    logger.info(f"--> New best selection metric ({selection_metric}): {best_metric:.4f}")
                if config.logging.use_wandb:
                    import wandb
                    payload = {
                        "epoch": epoch,
                        **{f"val/{k}": v for k, v in val_stats.items()},
                        "val/best_selection_metric": best_metric,
                    }
                    if hard_val_stats:
                        payload.update({f"hard_val/{k}": v for k, v in hard_val_stats.items()})
                    wandb.log(payload, step=dino_scheduler.current_step)

            if is_distributed and train_strategy == 'fsdp':
                is_best_tensor = torch.tensor([1 if is_best else 0], device=gpu, dtype=torch.int32)
                best_loss_tensor = torch.tensor([best_metric], device=gpu, dtype=torch.float32)
                dist.broadcast(is_best_tensor, src=0)
                dist.broadcast(best_loss_tensor, src=0)
                is_best = bool(is_best_tensor.item())
                best_metric = float(best_loss_tensor.item())

        should_save_epoch_checkpoint = (
            (epoch + 1) % config['logging']['save_freq'] == 0 or is_final_train_epoch
        )
        if should_save_epoch_checkpoint or is_best:
            checkpoint_state = get_unified_state_dict(
                model=model, optimizer=optimizer, scheduler=dino_scheduler, scaler=scaler,
                config=config, epoch=epoch, best_loss=best_metric, strategy=train_strategy  
            )
            checkpoint_state["best_metric"] = best_metric
            checkpoint_state["selection_metric"] = config['validation'].get('selection_metric', 'loss')
            checkpoint_state["model_metadata"] = model_metadata
            checkpoint_state["train_stats"] = train_stats
            checkpoint_state["val_stats"] = val_stats
            checkpoint_state["hard_val_stats"] = hard_val_stats
            save_checkpoint(
                checkpoint_state,
                checkpoint_dir,
                epoch,
                is_best,
                rank,
                train_strategy,
                save_epoch_checkpoint=should_save_epoch_checkpoint,
            )

        if reached_max_steps and bool(config['training'].get('stop_on_max_steps', True)):
            if rank == 0:
                logger.info(f"Reached training.max_steps={config['training']['max_steps']} at epoch {epoch}.")
            break

    if rank == 0:
        total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        logger.info(f"Training completed in {total_time_str} | Best Selection Metric: {best_metric:.4f} | Final LR: {optimizer.param_groups[0]['lr']:.6f}")

    cleanup_distributed()


if __name__ == '__main__':
    main()
