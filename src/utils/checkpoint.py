import torch
import os
import logging
import shutil
import re
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, LocalStateDictConfig

logger = logging.getLogger("neurojepa")


def _atomic_torch_save(obj, path):
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _atomic_copyfile(src, dst):
    tmp_path = f"{dst}.tmp.{os.getpid()}"
    try:
        shutil.copyfile(src, tmp_path)
        os.replace(tmp_path, dst)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

def _log_key_details(title, keys, max_items=200):
    if not keys:
        return
    keys = sorted(set(keys))
    logger.info("%s details (%d):", title, len(keys))
    for k in keys[:max_items]:
        logger.info("  - %s", k)
    if len(keys) > max_items:
        logger.info("  ... (%d more)", len(keys) - max_items)


def _resolve_fsdp_checkpoint_path(checkpoint_path):
    """
    Resolve per-rank local-shard checkpoint path for FSDP resume.
    If user passes ..._rank_0.pth under torchrun, each process will map it to
    ..._rank_{current_rank}.pth when available.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return checkpoint_path

    rank = dist.get_rank()
    dirname, basename = os.path.split(checkpoint_path)
    m = re.search(r"_rank_(\d+)\.pth$", basename)
    if m is None:
        return checkpoint_path

    name_with_rank = re.sub(r"_rank_\d+\.pth$", f"_rank_{rank}.pth", basename)
    resolved_path = os.path.join(dirname, name_with_rank)
    if os.path.isfile(resolved_path):
        return resolved_path
    return checkpoint_path


def _is_rank_shard_checkpoint(checkpoint_path):
    return re.search(r"_rank_\d+\.pth$", os.path.basename(checkpoint_path)) is not None


def _find_merged_checkpoint_for_shard(checkpoint_path):
    dirname, basename = os.path.split(checkpoint_path)

    if basename.startswith("checkpoint_best_rank_"):
        candidate = os.path.join(dirname, "merged_single_best.pth")
        return candidate if os.path.isfile(candidate) else None

    match = re.search(r"checkpoint_epoch_(\d+)_rank_\d+\.pth$", basename)
    if match:
        epoch = match.group(1)
        candidate = os.path.join(dirname, f"merged_single_epoch_{epoch}.pth")
        return candidate if os.path.isfile(candidate) else None

    return None


def _extract_model_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        if "model_state_dict" in checkpoint_obj:
            return checkpoint_obj["model_state_dict"]
        if "state_dict" in checkpoint_obj:
            return checkpoint_obj["state_dict"]
    return checkpoint_obj


def _load_checkpoint_file(checkpoint_path, train_strategy="ddp"):
    resolved_path = checkpoint_path
    if train_strategy == "fsdp" or _is_rank_shard_checkpoint(checkpoint_path):
        resolved_path = _resolve_fsdp_checkpoint_path(checkpoint_path)

    if _is_rank_shard_checkpoint(resolved_path) and not (dist.is_available() and dist.is_initialized()):
        merged = _find_merged_checkpoint_for_shard(resolved_path)
        if merged is not None:
            logger.warning(
                "Distributed not initialized; using merged checkpoint instead of rank shard: %s", merged
            )
            resolved_path = merged

    try:
        checkpoint_obj = torch.load(resolved_path, map_location="cpu", weights_only=False)
    except RuntimeError as e:
        err = str(e)
        mismatch = ("Local world size at save time" in err) or ("Local rank at save time" in err)
        need_pg = ("init_process_group" in err) and ("ShardedTensor" in err)
        if not mismatch and not need_pg:
            raise
        merged = _find_merged_checkpoint_for_shard(resolved_path)
        if merged is None:
            raise RuntimeError(
                "Failed to load rank-sharded FSDP checkpoint due rank/world-size mismatch or missing "
                "distributed process group, and "
                "no merged checkpoint was found. Please run merge_fsdp_checkpoint.py first."
            ) from e
        logger.warning(
            "Rank/world-size mismatch while loading shard; falling back to merged checkpoint: %s", merged
        )
        checkpoint_obj = torch.load(merged, map_location="cpu", weights_only=False)
        resolved_path = merged
    return checkpoint_obj, resolved_path


def get_unified_state_dict(model, optimizer, scheduler, scaler, config, epoch, best_loss, strategy):
    state = {
        'epoch': epoch + 1,
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': best_loss,
        'config': config,
    }
    if scaler is not None:
        state['scaler_state_dict'] = scaler.state_dict()

    if strategy == 'fsdp':
        local_save_policy = LocalStateDictConfig(offload_to_cpu=False)
        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT, local_save_policy):
            state['model_state_dict'] = model.state_dict()
            state['optimizer_state_dict'] = FSDP.optim_state_dict(model, optimizer)
    else:
        model_noddp = model.module if hasattr(model, 'module') else model
        state['model_state_dict'] = model_noddp.state_dict()
        state['optimizer_state_dict'] = optimizer.state_dict()

    return state


def save_checkpoint(state_dict, output_dir, epoch, is_best, rank, strategy=None, filename=None, save_epoch_checkpoint=True):

    if strategy != 'fsdp' and rank != 0:
        return

    save_path = None
    if save_epoch_checkpoint:
        if filename is None:
            if strategy == 'fsdp':
                filename = f'checkpoint_epoch_{epoch}_rank_{rank}.pth'
            else:
                filename = f'checkpoint_epoch_{epoch}.pth'
        save_path = os.path.join(output_dir, filename)
        _atomic_torch_save(state_dict, save_path)
        logger.info(f"Saved checkpoint: {filename}")

    if is_best:
        if strategy == 'fsdp':
            best_filename = f'checkpoint_best_rank_{rank}.pth'
        else:
            best_filename = 'checkpoint_best.pth'

        best_path = os.path.join(output_dir, best_filename)
        if save_path is not None:
            _atomic_copyfile(save_path, best_path)
        else:
            _atomic_torch_save(state_dict, best_path)
        if rank == 0:
            logger.info(f"Saved best checkpoint: {best_filename} (and other shards)")

def _normalize_pretrained_state_dict(state_dict, target="full"):
    normalized = {}
    for k, v in state_dict.items():
        key = k.replace("_fsdp_wrapped_module.", "")
        if key.startswith("module."):
            key = key[7:]

        # Skip non-model branches for self-supervised teacher/student checkpoints.
        if key.startswith("student.") and not key.startswith("student.backbone."):
            continue
        if key.startswith("teacher.") or key.startswith("dino_head."):
            continue

        if target == "backbone":
            if key.startswith("student.backbone."):
                key = key.replace("student.backbone.", "")
            elif key.startswith("backbone."):
                key = key.replace("backbone.", "")
            elif key.startswith("encoder."):
                key = key.replace("encoder.", "")

        if key.startswith("head"):
            continue

        # FSDP local shards may include sharded flat params; skip them for plain modules.
        if key.endswith("._flat_param"):
            continue
        if not torch.is_tensor(v):
            continue

        normalized[key] = v
    return normalized


def _is_bias_key(key):
    return key == "bias" or key.endswith(".bias")


def _adapt_pos_embed_if_needed(key, value, model_ref, model_state):
    if key != "pos_embed":
        return value, None
    if value.ndim != 3 or key not in model_state or model_state[key].ndim != 3:
        return value, None

    target = model_state[key]
    if value.shape[0] != target.shape[0] or value.shape[2] != target.shape[2]:
        return value, None
    if value.shape[1] == target.shape[1]:
        return value, None

    num_patches = getattr(model_ref, "num_patches", None)
    num_prefix_tokens = getattr(model_ref, "num_prefix_tokens", None)
    if num_patches is None or num_prefix_tokens is None:
        return value, None

    ckpt_prefix_tokens = int(value.shape[1]) - int(num_patches)
    if ckpt_prefix_tokens < 0:
        return value, None

    adapted = target.detach().clone()
    shared_prefix = min(int(num_prefix_tokens), ckpt_prefix_tokens)
    if shared_prefix > 0:
        adapted[:, :shared_prefix] = value[:, :shared_prefix]

    ckpt_patch_start = ckpt_prefix_tokens
    target_patch_start = int(num_prefix_tokens)
    if value.shape[1] - ckpt_patch_start != int(num_patches):
        return value, None
    adapted[:, target_patch_start:target_patch_start + int(num_patches)] = value[:, ckpt_patch_start:]
    detail = f"{tuple(value.shape)} -> {tuple(target.shape)}"
    return adapted, detail


def _load_pretrained_weights(checkpoint_obj, model, target="full", strict=False, skip_bias=False):
    raw_state = _extract_model_state_dict(checkpoint_obj)
    normalized_state = _normalize_pretrained_state_dict(raw_state, target=target)

    model_ref = model.module if hasattr(model, "module") else model
    model_state = model_ref.state_dict()

    filtered_state = {}
    skipped_shape = []
    skipped_bias = []
    adapted_keys = []
    for k, v in normalized_state.items():
        if k not in model_state:
            continue
        if skip_bias and _is_bias_key(k):
            skipped_bias.append(k)
            continue
        adapted_v, adapt_detail = _adapt_pos_embed_if_needed(k, v, model_ref, model_state)
        if adapt_detail is not None:
            v = adapted_v
            adapted_keys.append(f"{k}: {adapt_detail}")
        if model_state[k].shape != v.shape:
            skipped_shape.append(k)
            continue
        filtered_state[k] = v

    msg = model_ref.load_state_dict(filtered_state, strict=strict)
    logger.info("Pretrained weights loaded.")
    logger.info("Loaded tensors: %d", len(filtered_state))
    logger.info("Missing keys: %d", len(msg.missing_keys))
    logger.info("Unexpected keys: %d", len(msg.unexpected_keys))
    _log_key_details("Missing keys", msg.missing_keys)
    _log_key_details("Unexpected keys", msg.unexpected_keys)
    if skipped_bias:
        logger.info("Skipped (bias by config): %d", len(skipped_bias))
        _log_key_details("Skipped (bias by config)", skipped_bias)
    if skipped_shape:
        logger.info("Skipped (shape mismatch): %d", len(skipped_shape))
        _log_key_details("Skipped (shape mismatch)", skipped_shape)
    if adapted_keys:
        logger.info("Adapted checkpoint tensors: %d", len(adapted_keys))
        _log_key_details("Adapted checkpoint tensors", adapted_keys)
    return model


def load_checkpoint(
    checkpoint_path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    train_strategy='ddp',
    mode='resume',
    target='full',
    strict=False,
    skip_bias=False,
):
    """
    Unified checkpoint loader.

    mode='resume':
        Load model + optimizer + scheduler (+scaler) to continue training.
        Returns: (start_epoch, best_metric, best_loss)

    mode='pretrained':
        Load model weights only (for initialization / transfer learning).
        skip_bias=True will ignore all keys ending with ".bias" (and top-level "bias").
        Returns: model
    """
    if not os.path.isfile(checkpoint_path):
        logger.warning("No checkpoint found at '%s'", checkpoint_path)
        if mode == "resume":
            return 0, 0.0, 0.0
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'")

    if mode not in {"resume", "pretrained"}:
        raise ValueError(f"Unsupported mode: {mode}")

    if str(checkpoint_path).endswith('.safetensors'):
        if mode != 'pretrained':
            raise ValueError('Encoder-only safetensors cannot resume optimizer/training state.')
        if skip_bias:
            raise ValueError('Released encoder weights must be loaded with skip_bias=False.')
        from neurojepa.pretrained import available_models, _sha256
        for spec in available_models().values():
            weight_info = spec.get('weights')
            if weight_info and os.path.basename(checkpoint_path) == weight_info['filename']:
                if _sha256(checkpoint_path) != weight_info['sha256']:
                    raise ValueError('Checkpoint SHA-256 mismatch for named release weights.')
        from safetensors.torch import load_file
        state = load_file(str(checkpoint_path), device='cpu')
        model_to_load = model.module if hasattr(model, 'module') else model
        if target == 'backbone' and hasattr(model_to_load, 'backbone'):
            model_to_load = model_to_load.backbone
        expected = set(model_to_load.state_dict())
        missing = expected - set(state)
        unexpected = set(state) - expected
        # Only a newly initialized task head may be absent from encoder-only weights.
        bad_missing = {k for k in missing if not k.startswith('head.')}
        if bad_missing or unexpected:
            raise RuntimeError(f'Encoder key mismatch: missing={sorted(bad_missing)}, unexpected={sorted(unexpected)}')
        model_to_load.load_state_dict(state, strict=not missing)
        return model

    if mode == "pretrained":
        checkpoint_obj, used_path = _load_checkpoint_file(checkpoint_path, train_strategy=train_strategy)
        logger.info("Loading pretrained checkpoint from: %s", used_path)
        return _load_pretrained_weights(
            checkpoint_obj,
            model,
            target=target,
            strict=strict,
            skip_bias=skip_bias,
        )

    # mode == 'resume'
    if optimizer is None or scheduler is None:
        raise ValueError("optimizer and scheduler are required when mode='resume'")

    checkpoint_obj, used_path = _load_checkpoint_file(checkpoint_path, train_strategy=train_strategy)
    logger.info("Loading resume checkpoint from: %s", used_path)

    start_epoch = checkpoint_obj['epoch']
    best_metric = checkpoint_obj.get('best_metric', 0.0)
    best_loss = checkpoint_obj.get('best_loss', float('inf'))

    if train_strategy == 'fsdp':
        local_load_policy = LocalStateDictConfig(offload_to_cpu=False)
        model_state_dict = checkpoint_obj['model_state_dict']
        try:
            with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT, local_load_policy):
                incompat = model.load_state_dict(model_state_dict, strict=False)
        except AssertionError as e:
            if "No `FlatParameter` in `state_dict`" not in str(e):
                raise
            incompat = model.load_state_dict(model_state_dict, strict=False)

        unexpected = getattr(incompat, "unexpected_keys", [])
        missing = getattr(incompat, "missing_keys", [])
        if unexpected:
            logger.info("[FSDP resume] unexpected keys ignored: %d", len(unexpected))
        if missing:
            logger.info("[FSDP resume] missing keys ignored: %d", len(missing))

        local_optim_state = checkpoint_obj['optimizer_state_dict']
        sharded_optim_state = FSDP.optim_state_dict_to_load(
            model=model,
            optim=optimizer,
            optim_state_dict=local_optim_state
        )
        optimizer.load_state_dict(sharded_optim_state)
    else:
        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(checkpoint_obj['model_state_dict'])
        optimizer.load_state_dict(checkpoint_obj['optimizer_state_dict'])

    scheduler.load_state_dict(checkpoint_obj['scheduler_state_dict'])

    if scaler is not None and 'scaler_state_dict' in checkpoint_obj:
        scaler.load_state_dict(checkpoint_obj['scaler_state_dict'])

    logger.info("Loaded checkpoint from epoch %d", start_epoch)
    return start_epoch, best_metric, best_loss
