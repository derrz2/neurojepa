import argparse
import copy
import datetime
import hashlib
import json
import re
from pathlib import Path

import logging
logger = logging.getLogger("neurojepa")

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, LocalStateDictConfig
from torch.utils.data import DataLoader, TensorDataset, Subset

from src.models import create_model
from src.models.backbone import build_backbone
from src.data import create_downstream_dataloaders
from src.distributed.dist_ddp import setup_distributed, cleanup_distributed
from src.distributed.fsdp_helper import wrap_submodule_fsdp
from src.eval import evaluate
from src.train import finetune_one_epoch
from src.utils import set_seed, create_downstream_optimizer, create_downstream_scheduler
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.logging_utils import load_config, save_config, setup_logger
from src.utils.utils import LabelScaler


LP_GRID_PROTOCOL_VERSION = "lp_grid_v2_20260724"
DEFAULT_CLASSIFICATION_C_GRID = [10.0 ** exponent for exponent in range(-5, 4)]
DEFAULT_REGRESSION_ALPHA_GRID = [10.0 ** exponent for exponent in range(-4, 6)]


def _to_plain_dict(obj):
    if isinstance(obj, dict):
        return {key: _to_plain_dict(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain_dict(value) for value in obj]
    return obj


def _init_wandb(config, output_dir, rank):
    if rank != 0 or not config['logging'].get('use_wandb', False):
        return None

    try:
        import wandb
    except ImportError:
        logger.warning("use_wandb=True but wandb is not installed. Continuing without wandb.")
        return None

    if not hasattr(wandb, "init"):
        logger.warning("use_wandb=True but imported wandb module does not expose init(). Continuing without wandb.")
        return None

    return wandb.init(
        project=config['logging'].get('wandb_project'),
        group=config['logging'].get('wandb_group'),
        entity=config['logging'].get('wandb_entity'),
        config=_to_plain_dict(config),
        dir=str(output_dir),
        name=Path(output_dir).name,
    )


def _log_wandb_metrics(metrics, step=None):
    try:
        import wandb
    except ImportError:
        return

    if not hasattr(wandb, "log"):
        return

    if getattr(wandb, "run", None) is None:
        return

    wandb.log(metrics, step=step)


def _get_mode(config):
    return str(config['experiment'].get('mode', 'full_finetune'))


def _get_train_regression_label_stats(train_dataset):
    if not hasattr(train_dataset, 'get_regression_label_stats'):
        raise AttributeError(
            f"Dataset {type(train_dataset).__name__} does not support regression label statistics."
        )
    return train_dataset.get_regression_label_stats()


def _save_runtime_config(config, output_dir):
    save_config(config, output_dir / 'config.yaml')


def _make_eval_loader(dataset, data_config):
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        drop_last=False,
    )
    if data_config['num_workers'] > 0:
        loader_kwargs['prefetch_factor'] = data_config.get('prefetch_factor', 2)
    return DataLoader(**loader_kwargs)


def _is_rank_shard_checkpoint(checkpoint_path):
    return checkpoint_path is not None and re.search(r"_rank_\d+\.pth$", str(checkpoint_path)) is not None


def _rank_checkpoint_path(checkpoint_path, rank):
    checkpoint_path = str(checkpoint_path)
    return re.sub(r"_rank_\d+\.pth$", f"_rank_{rank}.pth", checkpoint_path)


def _strip_prefix_state_dict(state_dict, prefix):
    stripped = {}
    prefix = str(prefix)
    for key, value in state_dict.items():
        clean_key = key.replace("._fsdp_wrapped_module", "").replace("_fsdp_wrapped_module.", "")
        if clean_key.startswith(prefix):
            stripped[clean_key[len(prefix):]] = value
    return stripped


def _load_fsdp_rank_sharded_pretrained(model, checkpoint_path, rank, prefix="backbone."):
    local_path = _rank_checkpoint_path(checkpoint_path, rank)
    if not Path(local_path).is_file():
        raise FileNotFoundError(local_path)
    checkpoint_obj = torch.load(local_path, map_location="cpu", weights_only=False)
    local_state = checkpoint_obj.get("model_state_dict", checkpoint_obj)
    local_state = _strip_prefix_state_dict(local_state, prefix)
    if not local_state:
        raise RuntimeError(f"No tensors with prefix {prefix!r} found in {local_path}")
    load_policy = LocalStateDictConfig(offload_to_cpu=False)
    with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT, load_policy):
        incompat = model.load_state_dict(local_state, strict=False)
    return local_path, incompat


def _get_runtime_device(gpu):
    if torch.cuda.is_available():
        return torch.device(f'cuda:{gpu}')
    return torch.device('cpu')


def _unpack_batch(batch):
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        samples, labels, _ = batch
        return samples, labels
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        return batch
    raise ValueError("Unexpected batch format. Expected (samples, labels) or (samples, labels, dataset_idx).")


@torch.no_grad()
def _extract_feature_cache(model, data_loader, feature_source, device):
    model.eval()

    feature_chunks = []
    label_chunks = []
    use_amp = device.type == 'cuda'

    for batch in data_loader:
        samples, labels = _unpack_batch(batch)
        samples = samples.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            if feature_source in {'backbone', 'extract_probe_features'}:
                if isinstance(model, FSDP):
                    features = model(samples, return_probe_features=True)
                else:
                    if not hasattr(model, 'extract_probe_features'):
                        raise AttributeError(f"Model {type(model).__name__} does not implement extract_probe_features().")
                    features = model.extract_probe_features(samples)
            elif feature_source in {'temporal_pool', 'dino_temporal_pool'}:
                backbone = getattr(model, 'backbone', model)
                tokens = backbone(samples, return_tokens=True).float()
                bsz, num_tokens, dim = tokens.shape
                roi_patches = int(samples.shape[1])
                if num_tokens % roi_patches != 0:
                    raise ValueError(
                        f"Token count {num_tokens} is not divisible by ROI channels {roi_patches} "
                        f"for feature_source={feature_source}."
                    )
                temporal_patches = num_tokens // roi_patches
                tokens = tokens.reshape(bsz, roi_patches, temporal_patches, dim)
                features = tokens.mean(dim=1).reshape(bsz, temporal_patches * dim)
            else:
                raise ValueError(f"Unsupported probe feature source: {feature_source}")

        feature_chunks.append(features.detach().float().cpu())
        label_chunks.append(labels.detach().cpu())

    return torch.cat(feature_chunks, dim=0), torch.cat(label_chunks, dim=0)


def _get_probe_sample_pooling(probe_cfg):
    pooling = probe_cfg.get('sample_pooling', probe_cfg.get('feature_pooling', 'none'))
    if pooling is None:
        return 'none'
    return str(pooling).strip().lower()


def _get_probe_eval_pooling(probe_cfg):
    pooling = probe_cfg.get('eval_pooling', probe_cfg.get('prediction_pooling', 'none'))
    if pooling is None:
        return 'none'
    pooling = str(pooling).strip().lower()
    valid_poolings = {'none', 'off', 'disabled', 'subject_logit_mean', 'subject_logits_mean', 'subject_prediction_mean', 'mean_by_subject'}
    if pooling not in valid_poolings:
        raise ValueError(f"Unsupported probe.eval_pooling: {pooling}")
    return pooling


def _get_probe_eval_split_group(probe_cfg):
    group = probe_cfg.get('eval_split_group', probe_cfg.get('split_group', 'none'))
    if group is None:
        return 'none'
    group = str(group).strip().lower()
    valid_groups = {'none', 'off', 'disabled', 'subject', 'subjects', 'fixed'}
    if group not in valid_groups:
        raise ValueError(f"Unsupported probe.eval_split_group: {group}")
    if group in {'subject', 'subjects'}:
        return 'subject'
    if group == 'fixed':
        return 'fixed'
    return 'none'


def _uses_subject_eval_pooling(eval_pooling):
    return eval_pooling in {'subject_logit_mean', 'subject_logits_mean', 'subject_prediction_mean', 'mean_by_subject'}


def _ordered_unique_group_keys(group_keys):
    ordered = []
    seen = set()
    for key in group_keys:
        key = str(key)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def _summarize_group_keys(group_keys):
    grouped_counts = {}
    for key in group_keys:
        grouped_counts[str(key)] = grouped_counts.get(str(key), 0) + 1
    counts = list(grouped_counts.values())
    if not counts:
        return {
            'num_samples': 0,
            'num_groups': 0,
            'mean_group_size': 0.0,
            'min_group_size': 0,
            'max_group_size': 0,
        }
    return {
        'num_samples': len(group_keys),
        'num_groups': len(grouped_counts),
        'mean_group_size': float(sum(counts)) / float(len(counts)),
        'min_group_size': min(counts),
        'max_group_size': max(counts),
    }


def _build_subject_keys_from_dataset(dataset):
    if not hasattr(dataset, 'file_list'):
        raise AttributeError(f"Dataset {type(dataset).__name__} does not expose file_list for subject-level pooling.")
    if not hasattr(dataset, 'extract_subject_id'):
        raise AttributeError(f"Dataset {type(dataset).__name__} does not implement extract_subject_id for subject-level pooling.")

    subject_keys = []
    for item in dataset.file_list:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            raise ValueError("Expected dataset.file_list entries to have the form (dataset_name, file_path, label).")
        dataset_name, file_path, _ = item[:3]
        subject_id = dataset.extract_subject_id(dataset_name, file_path)
        if not subject_id:
            raise ValueError(
                f"Failed to parse subject id from '{file_path}' for dataset '{dataset_name}' during subject-level pooling."
            )
        subject_keys.append(f"{dataset_name}::{subject_id}")
    return subject_keys


def _aggregate_features_by_group(features, labels, group_keys, task_type):
    if len(group_keys) != int(features.shape[0]) or int(labels.shape[0]) != int(features.shape[0]):
        raise ValueError("Features, labels, and group keys must have the same number of samples.")

    grouped_indices = {}
    for idx, key in enumerate(group_keys):
        grouped_indices.setdefault(str(key), []).append(idx)

    aggregated_features = []
    aggregated_labels = []
    group_sizes = []

    for key, indices in grouped_indices.items():
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        group_features = features.index_select(0, idx_tensor)
        group_labels = labels.index_select(0, idx_tensor)

        aggregated_features.append(group_features.mean(dim=0))
        group_sizes.append(len(indices))

        if task_type == 'classification':
            label_values = group_labels.view(-1)
            reference_label = label_values[0]
            if not torch.equal(label_values, torch.full_like(label_values, reference_label)):
                raise ValueError(f"Classification labels disagree within pooled subject '{key}'.")
            aggregated_labels.append(reference_label.view(1))
        elif task_type == 'multilabel_classification':
            aggregated_labels.append(group_labels.float().mean(dim=0))
        else:
            aggregated_labels.append(group_labels.float().mean(dim=0, keepdim=False).view(1))

    return (
        torch.stack(aggregated_features, dim=0),
        torch.stack(aggregated_labels, dim=0) if task_type == 'multilabel_classification' else torch.cat(aggregated_labels, dim=0),
        {
            'num_groups': len(grouped_indices),
            'min_group_size': min(group_sizes),
            'max_group_size': max(group_sizes),
            'mean_group_size': float(sum(group_sizes)) / float(len(group_sizes)),
        },
    )


def _maybe_pool_probe_samples(dataset, split_name, features, labels, task_type, probe_cfg, rank, group_keys=None):
    sample_pooling = _get_probe_sample_pooling(probe_cfg)
    if sample_pooling in {'none', 'off', 'disabled'}:
        return features, labels, group_keys
    if sample_pooling not in {'subject_mean', 'subject-mean', 'mean_by_subject'}:
        raise ValueError(f"Unsupported probe.sample_pooling: {sample_pooling}")

    subject_keys = group_keys if group_keys is not None else _build_subject_keys_from_dataset(dataset)
    pooled_features, pooled_labels, pool_stats = _aggregate_features_by_group(
        features=features,
        labels=labels,
        group_keys=subject_keys,
        task_type=task_type,
    )

    if rank == 0:
        logger.info(
            f"{split_name}: subject-mean pooled {len(subject_keys)} crop samples into "
            f"{pool_stats['num_groups']} subjects "
            f"(mean_group_size={pool_stats['mean_group_size']:.2f}, "
            f"min={pool_stats['min_group_size']}, max={pool_stats['max_group_size']})."
        )

    return pooled_features, pooled_labels, _ordered_unique_group_keys(subject_keys)


def _split_eval_pool(num_samples, seed):
    if num_samples < 2:
        raise ValueError("Merged val/test pool must contain at least 2 samples.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_samples)
    mid = num_samples // 2
    val_idx = perm[:mid]
    test_idx = perm[mid:]
    if len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Split failed: val/test must both be non-empty.")
    return val_idx, test_idx


def _split_eval_pool_by_group(group_keys, seed):
    ordered_groups = _ordered_unique_group_keys(group_keys)
    if len(ordered_groups) < 2:
        raise ValueError("Merged val/test pool must contain at least 2 subject groups.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ordered_groups))
    mid = len(ordered_groups) // 2
    val_groups = {ordered_groups[idx] for idx in perm[:mid]}
    test_groups = {ordered_groups[idx] for idx in perm[mid:]}
    if len(val_groups) == 0 or len(test_groups) == 0:
        raise ValueError("Grouped split failed: val/test must both be non-empty.")

    val_idx = np.asarray([idx for idx, key in enumerate(group_keys) if key in val_groups], dtype=np.int64)
    test_idx = np.asarray([idx for idx, key in enumerate(group_keys) if key in test_groups], dtype=np.int64)
    return val_idx, test_idx


def _to_numpy_labels(labels, task_type):
    if task_type == 'multilabel_classification':
        return labels.cpu().numpy().astype(np.float32, copy=False)
    labels = labels.view(-1).cpu().numpy()
    if task_type == 'classification':
        return labels.astype(np.int64, copy=False)
    return labels.astype(np.float32, copy=False)


def _compute_corr(y_true, y_pred):
    if y_true.size < 2:
        return 0.0
    if float(np.std(y_true)) < 1e-12 or float(np.std(y_pred)) < 1e-12:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _softmax_np(scores):
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / np.clip(exp_scores.sum(axis=1, keepdims=True), 1e-12, None)


def _sigmoid_np(scores):
    scores = np.asarray(scores, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(scores, -60.0, 60.0)))


def _probabilities_from_scores(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        p1 = _sigmoid_np(scores)
        return np.stack([1.0 - p1, p1], axis=1)
    if scores.ndim == 2 and scores.shape[1] == 1:
        p1 = _sigmoid_np(scores[:, 0])
        return np.stack([1.0 - p1, p1], axis=1)
    if (
        scores.ndim == 2
        and np.all(scores >= -1e-7)
        and np.all(scores <= 1.0 + 1e-7)
        and np.allclose(scores.sum(axis=1), 1.0, atol=1e-4)
    ):
        probs = scores
    else:
        probs = _softmax_np(scores)
    return probs


def _add_log_score_aliases(metrics, model_log_loss, null_log_loss):
    gain = float(null_log_loss - model_log_loss)
    metrics['model_log_loss'] = float(model_log_loss)
    metrics['log_loss'] = float(model_log_loss)
    metrics['null_log_loss'] = float(null_log_loss)
    metrics['log_score_gain'] = gain
    metrics['predictive_information_gain'] = gain
    metrics['log_likelihood_gain'] = gain


def _classification_log_score_metrics(y_true, y_score, null_targets=None):
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    probs = _probabilities_from_scores(y_score)
    n_classes = int(probs.shape[1])
    valid = (y_true >= 0) & (y_true < n_classes)
    if not np.any(valid):
        return None
    y_eval = y_true[valid]
    probs = np.clip(probs[valid], 1e-12, 1.0)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    model_log_loss = -float(np.mean(np.log(probs[np.arange(len(y_eval)), y_eval])))
    y_null = y_eval if null_targets is None else np.asarray(null_targets, dtype=np.int64).reshape(-1)
    y_null = y_null[(y_null >= 0) & (y_null < n_classes)]
    if y_null.size == 0:
        y_null = y_eval
    counts = np.bincount(y_null, minlength=n_classes).astype(np.float64)
    null_probs = np.clip(counts / np.clip(counts.sum(), 1e-12, None), 1e-12, 1.0)
    null_probs = null_probs / null_probs.sum()
    null_log_loss = -float(np.mean(np.log(null_probs[y_eval])))
    return model_log_loss, null_log_loss


def _multilabel_log_score_metrics(y_true, y_score, null_targets=None):
    y_true = np.asarray(y_true, dtype=np.float64)
    probs = np.clip(_sigmoid_np(y_score), 1e-12, 1.0 - 1e-12)
    if probs.shape != y_true.shape:
        return None
    model_log_loss = -float(np.mean(y_true * np.log(probs) + (1.0 - y_true) * np.log(1.0 - probs)))
    y_null = y_true if null_targets is None else np.asarray(null_targets, dtype=np.float64)
    if y_null.ndim == 1:
        y_null = y_null.reshape(-1, 1)
    if y_null.shape[1:] != y_true.shape[1:]:
        y_null = y_true
    null_probs = np.clip(y_null.mean(axis=0, keepdims=True), 1e-12, 1.0 - 1e-12)
    null_log_loss = -float(np.mean(y_true * np.log(null_probs) + (1.0 - y_true) * np.log(1.0 - null_probs)))
    return model_log_loss, null_log_loss


def _regression_log_score_metrics(y_true, y_pred, null_targets=None):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return None
    model_mse = float(np.mean((y_true - y_pred) ** 2))
    y_null = y_true if null_targets is None else np.asarray(null_targets, dtype=np.float64).reshape(-1)
    null_mean = float(np.mean(y_null)) if y_null.size else float(np.mean(y_true))
    null_mse = float(np.mean((y_true - null_mean) ** 2))
    model_var = max(model_mse, 1e-12)
    null_var = max(null_mse, 1e-12)
    model_log_loss = 0.5 * (1.0 + np.log(2.0 * np.pi * model_var))
    null_log_loss = 0.5 * (1.0 + np.log(2.0 * np.pi * null_var))
    return float(model_log_loss), float(null_log_loss)


def _compute_probe_metrics(task_type, y_true, y_pred, y_score=None, null_targets=None):
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )

    if task_type == 'classification':
        acc = float(accuracy_score(y_true, y_pred))
        metrics = {
            'loss': float(1.0 - acc),
            'acc': acc,
            'f1': float(f1_score(y_true, y_pred, average='weighted')),
        }
        if y_score is not None:
            log_metrics = _classification_log_score_metrics(y_true, y_score, null_targets=null_targets)
            if log_metrics is not None:
                _add_log_score_aliases(metrics, *log_metrics)
        return metrics

    if task_type == 'multilabel_classification':
        y_true = np.asarray(y_true, dtype=np.float32)
        y_score = np.asarray(y_pred if y_score is None else y_score, dtype=np.float32)
        y_hat = (y_score >= 0.0).astype(np.int64)
        metrics = {
            'loss': float(1.0 - f1_score(y_true, y_hat, average='micro', zero_division=0)),
            'micro_f1': float(f1_score(y_true, y_hat, average='micro', zero_division=0)),
            'macro_f1': float(f1_score(y_true, y_hat, average='macro', zero_division=0)),
            'label_cardinality_error': float(np.mean(np.abs(y_true.sum(axis=1) - y_hat.sum(axis=1)))),
        }
        try:
            metrics['micro_map'] = float(average_precision_score(y_true, y_score, average='micro'))
            metrics['macro_map'] = float(average_precision_score(y_true, y_score, average='macro'))
        except ValueError:
            metrics['micro_map'] = 0.0
            metrics['macro_map'] = 0.0
        try:
            metrics['micro_auroc'] = float(roc_auc_score(y_true, y_score, average='micro'))
            metrics['macro_auroc'] = float(roc_auc_score(y_true, y_score, average='macro'))
        except ValueError:
            metrics['micro_auroc'] = 0.0
            metrics['macro_auroc'] = 0.0
        log_metrics = _multilabel_log_score_metrics(y_true, y_score, null_targets=null_targets)
        if log_metrics is not None:
            _add_log_score_aliases(metrics, *log_metrics)
        return metrics

    mse = float(mean_squared_error(y_true, y_pred))
    metrics = {
        'loss': mse,
        'mse': mse,
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
        'corr': _compute_corr(y_true, y_pred),
    }
    log_metrics = _regression_log_score_metrics(y_true, y_pred, null_targets=null_targets)
    if log_metrics is not None:
        _add_log_score_aliases(metrics, *log_metrics)
    return metrics


def _pool_eval_outputs_by_group(outputs, labels, group_keys, task_type):
    if group_keys is None:
        return outputs, labels
    if len(group_keys) != int(outputs.shape[0]) or int(labels.shape[0]) != int(outputs.shape[0]):
        raise ValueError("Outputs, labels, and group keys must have the same number of samples.")

    grouped_indices = {}
    for idx, key in enumerate(group_keys):
        grouped_indices.setdefault(str(key), []).append(idx)

    pooled_outputs = []
    pooled_labels = []
    for key, indices in grouped_indices.items():
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        group_outputs = outputs.index_select(0, idx_tensor)
        group_labels = labels.index_select(0, idx_tensor)
        pooled_outputs.append(group_outputs.float().mean(dim=0))

        if task_type == 'classification':
            label_values = group_labels.view(-1)
            reference_label = label_values[0]
            if not torch.equal(label_values, torch.full_like(label_values, reference_label)):
                raise ValueError(f"Classification labels disagree within pooled subject '{key}'.")
            pooled_labels.append(reference_label.view(1))
        elif task_type == 'multilabel_classification':
            pooled_labels.append(group_labels.float().mean(dim=0))
        else:
            pooled_labels.append(group_labels.float().mean(dim=0, keepdim=False).view(1))

    label_out = torch.stack(pooled_labels, dim=0) if task_type == 'multilabel_classification' else torch.cat(pooled_labels, dim=0)
    return torch.stack(pooled_outputs, dim=0), label_out


def _compute_probe_metrics_from_outputs(task_type, outputs, labels, group_keys=None, null_targets=None):
    if group_keys is not None:
        outputs, labels = _pool_eval_outputs_by_group(outputs, labels, group_keys, task_type)

    if task_type == 'classification':
        if outputs.ndim == 1:
            preds = (outputs >= 0).long()
        elif outputs.ndim == 2 and outputs.shape[1] == 1:
            preds = (outputs.view(-1) >= 0).long()
        else:
            preds = outputs.argmax(dim=1)
    elif task_type == 'multilabel_classification':
        preds = outputs.float()
    else:
        preds = outputs.view(-1)

    y_pred = _to_numpy_labels(preds, task_type)
    y_true = _to_numpy_labels(labels if task_type == 'multilabel_classification' else labels.view(-1), task_type)
    y_score = outputs.detach().cpu().numpy() if torch.is_tensor(outputs) else np.asarray(outputs)
    y_null = None
    if null_targets is not None:
        null_tensor = null_targets if torch.is_tensor(null_targets) else torch.as_tensor(null_targets)
        y_null = _to_numpy_labels(null_tensor if task_type == 'multilabel_classification' else null_tensor.view(-1), task_type)
    return _compute_probe_metrics(task_type, y_true, y_pred, y_score=y_score, null_targets=y_null)


def _linear_probe_grid(task_type, probe_cfg):
    linear_cfg = probe_cfg.get('linear_probe', {})
    if task_type in {'classification', 'multilabel_classification'}:
        parameter_name = 'C'
        raw_grid = linear_cfg.get('C_grid', linear_cfg.get('c_grid', DEFAULT_CLASSIFICATION_C_GRID))
    else:
        parameter_name = 'alpha'
        raw_grid = linear_cfg.get('alpha_grid', DEFAULT_REGRESSION_ALPHA_GRID)

    if not isinstance(raw_grid, (list, tuple)) or not raw_grid:
        raise ValueError(f"probe.linear_probe.{parameter_name}_grid must be a non-empty list")

    grid = []
    for raw_value in raw_grid:
        value = float(raw_value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                f"probe.linear_probe.{parameter_name}_grid contains invalid value: {raw_value!r}"
            )
        if value not in grid:
            grid.append(value)
    return parameter_name, grid


def _build_probe_estimator(method, task_type, probe_cfg, seed, selected_hyperparameter=None):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC, LinearSVR

    if method == 'linear_probe':
        linear_cfg = probe_cfg.get('linear_probe', {})
        if task_type == 'classification':
            C = float(selected_hyperparameter if selected_hyperparameter is not None else linear_cfg.get('C', 1.0))
            estimator = LogisticRegression(
                C=C,
                max_iter=int(linear_cfg.get('max_iter', 2000)),
                random_state=seed,
            )
        elif task_type == 'multilabel_classification':
            C = float(selected_hyperparameter if selected_hyperparameter is not None else linear_cfg.get('C', 1.0))
            estimator = OneVsRestClassifier(
                LogisticRegression(
                    C=C,
                    max_iter=int(linear_cfg.get('max_iter', 2000)),
                    random_state=seed,
                )
            )
        else:
            alpha = float(
                selected_hyperparameter if selected_hyperparameter is not None else linear_cfg.get('alpha', 1.0)
            )
            estimator = Ridge(alpha=alpha)
    elif method == 'ridge':
        if task_type != 'regression':
            raise ValueError("ridge mode only supports regression tasks.")
        ridge_cfg = probe_cfg.get('ridge', {})
        linear_cfg = probe_cfg.get('linear_probe', {})
        estimator = Ridge(alpha=float(ridge_cfg.get('alpha', linear_cfg.get('alpha', 1.0))))
    elif method == 'svm':
        svm_cfg = probe_cfg.get('svm', {})
        if str(svm_cfg.get('kernel', 'linear')) != 'linear':
            raise ValueError("Only linear SVM is supported in feature-based probe mode.")
        if task_type == 'classification':
            estimator = LinearSVC(
                C=float(svm_cfg.get('C', 1.0)),
                max_iter=int(svm_cfg.get('max_iter', 5000)),
                random_state=seed,
            )
        elif task_type == 'multilabel_classification':
            estimator = OneVsRestClassifier(
                LinearSVC(
                    C=float(svm_cfg.get('C', 1.0)),
                    max_iter=int(svm_cfg.get('max_iter', 5000)),
                    random_state=seed,
                )
            )
        else:
            estimator = LinearSVR(
                C=float(svm_cfg.get('C', 1.0)),
                max_iter=int(svm_cfg.get('max_iter', 5000)),
                random_state=seed,
            )
    else:
        raise ValueError(f"Unsupported feature-based method: {method}")

    return make_pipeline(StandardScaler(), estimator)


def _predict_probe_outputs(estimator, task_type, features):
    if task_type in {'classification', 'multilabel_classification'}:
        if hasattr(estimator, 'decision_function'):
            outputs = estimator.decision_function(features)
        elif hasattr(estimator, 'predict_proba'):
            outputs = estimator.predict_proba(features)
        else:
            outputs = estimator.predict(features)
    else:
        outputs = estimator.predict(features)
    return torch.as_tensor(outputs)


def _fit_and_evaluate_probe(
    method,
    task_type,
    probe_cfg,
    seed,
    train_X,
    train_y,
    val_X,
    val_y,
    test_X,
    test_y,
    val_group_keys=None,
    test_group_keys=None,
    *,
    return_estimator=False,
):
    eval_pooling = _get_probe_eval_pooling(probe_cfg)

    train_targets = train_y
    val_targets = val_y
    test_targets = test_y
    label_mean = None
    label_std = None
    if task_type == 'regression':
        label_mean = float(train_y.mean())
        label_std = float(train_y.std())
        if label_std < 1e-8:
            label_std = 1.0
        train_targets = (train_y - label_mean) / label_std
        val_targets = (val_y - label_mean) / label_std
        test_targets = (test_y - label_mean) / label_std

    pooled_val_group_keys = val_group_keys if _uses_subject_eval_pooling(eval_pooling) else None
    pooled_test_group_keys = test_group_keys if _uses_subject_eval_pooling(eval_pooling) else None

    parameter_name = None
    grid = [None]
    if method == 'linear_probe':
        parameter_name, grid = _linear_probe_grid(task_type, probe_cfg)

    best_estimator = None
    best_val_stats = None
    best_value = None
    best_loss = float('inf')
    grid_results = []

    for candidate in grid:
        try:
            estimator = _build_probe_estimator(
                method,
                task_type,
                probe_cfg,
                seed,
                selected_hyperparameter=candidate,
            )
            estimator.fit(train_X, train_targets)
            val_outputs = _predict_probe_outputs(estimator, task_type, val_X)
            val_stats = _compute_probe_metrics_from_outputs(
                task_type,
                val_outputs,
                torch.as_tensor(val_targets),
                group_keys=pooled_val_group_keys,
                null_targets=torch.as_tensor(train_targets),
            )
            loss = float(val_stats['loss'])
            if not np.isfinite(loss):
                raise ValueError(f"validation loss is not finite: {loss}")
            if parameter_name is not None:
                grid_results.append({
                    parameter_name: float(candidate),
                    'val_stats': _to_plain_dict(val_stats),
                })
            if best_estimator is None or loss < best_loss:
                best_estimator = estimator
                best_val_stats = val_stats
                best_value = candidate
                best_loss = loss
        except Exception as exc:
            if parameter_name is None:
                raise
            grid_results.append({
                parameter_name: float(candidate),
                'error': repr(exc),
            })

    if best_estimator is None:
        raise RuntimeError(
            f"All {parameter_name or 'probe'} candidates failed for seed={seed}: {grid_results}"
        )

    estimator = best_estimator
    test_outputs = _predict_probe_outputs(estimator, task_type, test_X)

    test_stats = _compute_probe_metrics_from_outputs(
        task_type,
        test_outputs,
        torch.as_tensor(test_targets),
        group_keys=pooled_test_group_keys,
        null_targets=torch.as_tensor(train_targets),
    )
    selection = None
    if parameter_name is not None:
        selection = {
            'protocol_version': LP_GRID_PROTOCOL_VERSION,
            'selection_metric': 'val_loss',
            'seed': int(seed),
            'grid_parameter': parameter_name,
            'grid': [float(value) for value in grid],
            f'selected_{parameter_name}': float(best_value),
            'grid_search_results': grid_results,
        }
        logger.info(
            "seed=%d linear-probe grid search selected_%s=%g from %s using validation loss %.6f",
            seed,
            parameter_name,
            best_value,
            grid,
            best_loss,
        )

    if return_estimator:
        # Reuse the selected train-fitted model in the cached-feature CLI;
        # keep the original runner's three-value return contract unchanged.
        return best_val_stats, test_stats, selection, estimator, test_outputs
    return best_val_stats, test_stats, selection


def _build_mlp_probe_module(input_dim, output_dim, mlp_cfg):
    hidden_dim = int(mlp_cfg.get('hidden_dim', 64))
    dropout = float(mlp_cfg.get('dropout', 0.1))
    num_layers = max(int(mlp_cfg.get('num_layers', 2)), 1)
    input_norm = str(mlp_cfg.get('input_norm', 'layernorm')).lower()

    layers = []
    if input_norm == 'layernorm':
        layers.append(nn.LayerNorm(int(input_dim)))
    elif input_norm not in {'none', 'identity'}:
        raise ValueError(f"Unsupported mlp_probe.input_norm: {input_norm}")

    if num_layers == 1:
        layers.append(nn.Linear(int(input_dim), int(output_dim)))
        return nn.Sequential(*layers)

    in_dim = int(input_dim)
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = hidden_dim

    layers.append(nn.Linear(in_dim, int(output_dim)))
    return nn.Sequential(*layers)


def _standardize_feature_tensors(train_X, *other_splits):
    feature_mean = train_X.mean(dim=0, keepdim=True)
    feature_std = train_X.std(dim=0, keepdim=True, unbiased=False)
    feature_std = torch.where(feature_std < 1e-6, torch.ones_like(feature_std), feature_std)

    standardized = [(train_X - feature_mean) / feature_std]
    standardized.extend((split - feature_mean) / feature_std for split in other_splits)
    return standardized, feature_mean.cpu(), feature_std.cpu()


def _make_feature_loader(features, targets, batch_size, shuffle, seed):
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    return DataLoader(
        TensorDataset(features, targets),
        batch_size=int(batch_size),
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


@torch.no_grad()
def _evaluate_mlp_probe(probe, task_type, features, targets, batch_size, device, use_amp, group_keys=None, eval_pooling='none', null_targets=None):
    probe.eval()
    loader = _make_feature_loader(features, targets, batch_size=batch_size, shuffle=False, seed=0)

    output_chunks = []
    target_chunks = []
    for feature_batch, target_batch in loader:
        feature_batch = feature_batch.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = probe(feature_batch)
        if task_type == 'regression':
            outputs = outputs.view(-1)
        output_chunks.append(outputs.detach().float().cpu())
        target_chunks.append(target_batch.detach().cpu())

    outputs = torch.cat(output_chunks, dim=0)
    targets = torch.cat(target_chunks, dim=0)
    pooled_group_keys = group_keys if _uses_subject_eval_pooling(eval_pooling) else None
    return _compute_probe_metrics_from_outputs(task_type, outputs, targets, group_keys=pooled_group_keys, null_targets=null_targets)


def _fit_and_evaluate_mlp_probe(
    task_type,
    num_classes,
    probe_cfg,
    seed,
    train_X,
    train_y,
    val_X,
    val_y,
    test_X,
    test_y,
    device,
    rank,
    val_group_keys=None,
    test_group_keys=None,
):
    mlp_cfg = probe_cfg.get('mlp_probe', {})
    eval_pooling = _get_probe_eval_pooling(probe_cfg)
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_X = train_X.float().contiguous()
    val_X = val_X.float().contiguous()
    test_X = test_X.float().contiguous()
    (train_X, val_X, test_X), feature_mean, feature_std = _standardize_feature_tensors(train_X, val_X, test_X)

    if task_type == 'classification':
        train_targets = train_y.view(-1).long().contiguous()
        val_targets = val_y.view(-1).long().contiguous()
        test_targets = test_y.view(-1).long().contiguous()
        output_dim = int(num_classes)
        criterion = nn.CrossEntropyLoss()
        label_mean = None
        label_std = None
    elif task_type == 'multilabel_classification':
        train_targets = train_y.float().contiguous()
        val_targets = val_y.float().contiguous()
        test_targets = test_y.float().contiguous()
        output_dim = int(num_classes)
        criterion = nn.BCEWithLogitsLoss()
        label_mean = None
        label_std = None
    else:
        train_targets = train_y.view(-1).float().contiguous()
        val_targets = val_y.view(-1).float().contiguous()
        test_targets = test_y.view(-1).float().contiguous()
        label_mean = float(train_targets.mean().item())
        label_std = float(train_targets.std(unbiased=False).item())
        if label_std < 1e-8:
            label_std = 1.0
        train_targets = (train_targets - label_mean) / label_std
        val_targets = (val_targets - label_mean) / label_std
        test_targets = (test_targets - label_mean) / label_std
        output_dim = 1
        criterion = nn.MSELoss()

    probe = _build_mlp_probe_module(train_X.shape[1], output_dim, mlp_cfg).to(device)

    optimizer_name = str(mlp_cfg.get('optimizer', 'adamw')).lower()
    learning_rate = float(mlp_cfg.get('learning_rate', 1.0e-3))
    weight_decay = float(mlp_cfg.get('weight_decay', 1.0e-6))
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            probe.parameters(),
            lr=learning_rate,
            momentum=float(mlp_cfg.get('momentum', 0.9)),
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported mlp_probe.optimizer: {optimizer_name}")

    train_batch_size = int(mlp_cfg.get('batch_size', 256))
    eval_batch_size = int(mlp_cfg.get('eval_batch_size', max(train_batch_size, 1024)))
    epochs = int(mlp_cfg.get('epochs', 100))
    warmup_epochs = int(mlp_cfg.get('warmup_epochs', 10))
    min_lr = float(mlp_cfg.get('min_lr', 1.0e-6))
    log_freq = int(mlp_cfg.get('log_freq', 10))
    use_amp = bool(mlp_cfg.get('use_amp', True)) and device.type == 'cuda'
    clip_grad = mlp_cfg.get('clip_grad', None)

    train_loader = _make_feature_loader(train_X, train_targets, batch_size=train_batch_size, shuffle=True, seed=seed)
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = max(epochs * steps_per_epoch, 1)
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(current_step):
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)

        if total_steps <= warmup_steps:
            return 1.0

        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return max(min_lr / learning_rate, cosine)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=use_amp)

    best_val_stats = None
    best_test_stats = None
    best_epoch = -1
    best_state_dict = None

    for epoch in range(epochs):
        probe.train()
        loss_sum = 0.0
        sample_count = 0

        for feature_batch, target_batch in train_loader:
            feature_batch = feature_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = probe(feature_batch)
                if task_type == 'classification':
                    loss = criterion(outputs, target_batch)
                elif task_type == 'multilabel_classification':
                    loss = criterion(outputs, target_batch)
                else:
                    loss = criterion(outputs.view_as(target_batch), target_batch)

            scaler.scale(loss).backward()
            if clip_grad is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(probe.parameters(), float(clip_grad))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            batch_size = int(feature_batch.shape[0])
            loss_sum += float(loss.item()) * batch_size
            sample_count += batch_size

        train_loss = loss_sum / max(sample_count, 1)
        val_stats = _evaluate_mlp_probe(
            probe,
            task_type,
            val_X,
            val_targets,
            batch_size=eval_batch_size,
            device=device,
            use_amp=use_amp,
            group_keys=val_group_keys,
            eval_pooling=eval_pooling,
            null_targets=train_targets,
        )
        test_stats = _evaluate_mlp_probe(
            probe,
            task_type,
            test_X,
            test_targets,
            batch_size=eval_batch_size,
            device=device,
            use_amp=use_amp,
            group_keys=test_group_keys,
            eval_pooling=eval_pooling,
            null_targets=train_targets,
        )

        if best_val_stats is None or val_stats['loss'] < best_val_stats['loss']:
            best_val_stats = val_stats
            best_test_stats = test_stats
            best_epoch = epoch
            best_state_dict = {k: v.detach().cpu() for k, v in probe.state_dict().items()}

        if rank == 0 and (epoch == 0 or (epoch + 1) % log_freq == 0 or epoch == epochs - 1):
            logger.info(
                f"mlp_probe epoch {epoch + 1}/{epochs} - "
                f"train_loss: {train_loss:.4f} | "
                + " | ".join([f"val_{k}: {v:.4f}" for k, v in val_stats.items()])
                + " | "
                + " | ".join([f"test_{k}: {v:.4f}" for k, v in test_stats.items()])
            )

    checkpoint_payload = {
        'probe_state_dict': best_state_dict,
        'probe_type': 'mlp_probe',
        'probe_input_dim': int(train_X.shape[1]),
        'feature_mean': feature_mean,
        'feature_std': feature_std,
    }
    if label_mean is not None:
        checkpoint_payload['label_mean'] = label_mean
        checkpoint_payload['label_std'] = label_std

    return best_val_stats, best_test_stats, best_epoch, checkpoint_payload


def _save_probe_run_artifact(run_dir, run_config, val_stats, test_stats, epoch=0, extra_checkpoint=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    save_config(run_config, run_dir / 'config.yaml')

    checkpoint = {
        'epoch': int(epoch),
        'best_loss': float(val_stats['loss']),
        'config': run_config,
        'val_stats': val_stats,
        'test_stats': test_stats,
    }
    if extra_checkpoint:
        checkpoint.update(extra_checkpoint)
    torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{int(epoch)}.pth')
    torch.save(checkpoint, checkpoint_dir / 'checkpoint_best.pth')


def _summarize_probe_results(results):
    summary = {}
    for split_name in ('val_stats', 'test_stats'):
        metric_names = sorted({key for result in results for key in result[split_name].keys()})
        split_summary = {}
        for metric_name in metric_names:
            values = torch.tensor(
                [result[split_name][metric_name] for result in results],
                dtype=torch.float32,
            )
            split_summary[metric_name] = {
                'mean': float(values.mean().item()),
                'std': float(values.std(unbiased=False).item()),
            }
        summary[split_name] = split_summary
    return summary


def _write_probe_summary(output_dir, method, results, probe_cfg=None):
    summary = {
        'mode': method,
        'num_runs': len(results),
        'runs': results,
        'summary': _summarize_probe_results(results),
    }
    if method == 'linear_probe':
        summary['protocol_version'] = (probe_cfg or {}).get(
            'protocol_version',
            LP_GRID_PROTOCOL_VERSION,
        )
    with open(output_dir / 'probe_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def _extract_distributed_feature_cache(model, dataset, data_cfg, feature_source, device, rank, world_size):
    group_keys = _build_subject_keys_from_dataset(dataset)
    local_indices = list(range(rank, len(dataset), world_size))
    local_dataset = Subset(dataset, local_indices)
    local_loader = _make_eval_loader(local_dataset, data_cfg)
    local_features, local_labels = _extract_feature_cache(model, local_loader, feature_source, device)
    local_group_keys = [group_keys[idx] for idx in local_indices]
    gathered = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object((local_features, local_labels, local_group_keys), gathered, dst=0)
    dist.barrier()
    if rank != 0:
        return None, None, None
    feature_parts = [item[0] for item in gathered if item is not None and item[0].numel() > 0]
    label_parts = [item[1] for item in gathered if item is not None and item[1].numel() > 0]
    merged_group_keys = []
    for item in gathered:
        if item is not None:
            merged_group_keys.extend(item[2])
    return torch.cat(feature_parts, dim=0), torch.cat(label_parts, dim=0), merged_group_keys


def run_feature_based_probe(config, output_dir, model, train_dataset, val_dataset, test_dataset, rank, gpu):
    method = _get_mode(config)
    distributed_probe = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    world_size = dist.get_world_size() if distributed_probe else 1

    probe_cfg = config.get('probe', {})
    feature_source = str(probe_cfg.get('feature_source', 'extract_probe_features'))
    task_type = config['task']['task_type']
    eval_pooling = _get_probe_eval_pooling(probe_cfg)
    eval_split_group = _get_probe_eval_split_group(probe_cfg)

    device = _get_runtime_device(gpu)
    model_ref = model if isinstance(model, FSDP) else (model.module if hasattr(model, 'module') else model)

    if rank == 0 and eval_split_group == 'subject':
        logger.info("Using subject-disjoint val/test resampling for each probe run without pooling outputs.")
    if rank == 0 and eval_split_group == 'fixed':
        logger.info("Using fixed manifest val/test split for each probe run.")
    if rank == 0 and _uses_subject_eval_pooling(eval_pooling):
        logger.info(
            "Using subject-level eval pooling on final probe outputs; "
            "val/test splits will be sampled by subject and outputs will be mean-aggregated per subject."
        )

    data_cfg = config['data']
    if not distributed_probe:
        train_feature_loader = _make_eval_loader(train_dataset, data_cfg)
        val_feature_loader = _make_eval_loader(val_dataset, data_cfg)
        test_feature_loader = _make_eval_loader(test_dataset, data_cfg)

    train_group_keys = _build_subject_keys_from_dataset(train_dataset)
    val_group_keys = _build_subject_keys_from_dataset(val_dataset)
    test_group_keys = _build_subject_keys_from_dataset(test_dataset)
    if rank == 0 and _uses_subject_eval_pooling(eval_pooling):
        for split_name, group_keys in (
            ('train', train_group_keys),
            ('val', val_group_keys),
            ('test', test_group_keys),
        ):
            group_stats = _summarize_group_keys(group_keys)
            logger.info(
                "%s subject grouping before feature extraction: samples=%d | subjects=%d | "
                "mean_group_size=%.2f | min=%d | max=%d",
                split_name,
                group_stats['num_samples'],
                group_stats['num_groups'],
                group_stats['mean_group_size'],
                group_stats['min_group_size'],
                group_stats['max_group_size'],
            )

    if rank == 0:
        logger.info("Extracting frozen features for train split...")
    if distributed_probe:
        train_features, train_labels, train_group_keys = _extract_distributed_feature_cache(
            model_ref, train_dataset, data_cfg, feature_source, device, rank, world_size
        )
    else:
        train_features, train_labels = _extract_feature_cache(model_ref, train_feature_loader, feature_source, device)
    if rank == 0:
        train_features, train_labels, train_group_keys = _maybe_pool_probe_samples(
            dataset=train_dataset,
            split_name='train',
            features=train_features,
            labels=train_labels,
            task_type=task_type,
            probe_cfg=probe_cfg,
            rank=rank,
            group_keys=train_group_keys,
        )

    if rank == 0:
        logger.info("Extracting frozen features for validation split...")
    if distributed_probe:
        val_features, val_labels, val_group_keys = _extract_distributed_feature_cache(
            model_ref, val_dataset, data_cfg, feature_source, device, rank, world_size
        )
    else:
        val_features, val_labels = _extract_feature_cache(model_ref, val_feature_loader, feature_source, device)
    if rank == 0:
        val_features, val_labels, val_group_keys = _maybe_pool_probe_samples(
            dataset=val_dataset,
            split_name='val',
            features=val_features,
            labels=val_labels,
            task_type=task_type,
            probe_cfg=probe_cfg,
            rank=rank,
            group_keys=val_group_keys,
        )

    if rank == 0:
        logger.info("Extracting frozen features for test split...")
    if distributed_probe:
        test_features, test_labels, test_group_keys = _extract_distributed_feature_cache(
            model_ref, test_dataset, data_cfg, feature_source, device, rank, world_size
        )
    else:
        test_features, test_labels = _extract_feature_cache(model_ref, test_feature_loader, feature_source, device)
    if rank == 0:
        test_features, test_labels, test_group_keys = _maybe_pool_probe_samples(
            dataset=test_dataset,
            split_name='test',
            features=test_features,
            labels=test_labels,
            task_type=task_type,
            probe_cfg=probe_cfg,
            rank=rank,
            group_keys=test_group_keys,
        )

    if distributed_probe and rank != 0:
        return

    eval_features = torch.cat([val_features, test_features], dim=0)
    eval_labels = torch.cat([val_labels, test_labels], dim=0)
    eval_group_keys = val_group_keys + test_group_keys
    train_X = train_features
    train_y = train_labels

    num_runs = int(probe_cfg.get('num_runs', 5))
    base_seed = int(probe_cfg.get('base_seed', config['experiment']['seed']))
    seed_stride = int(probe_cfg.get('seed_stride', 1))
    independent_validation_split = bool(
        probe_cfg.get('independent_validation_split_per_seed', False)
    )
    protocol_version = str(probe_cfg.get('protocol_version', ''))
    run_seeds = [base_seed + run_idx * seed_stride for run_idx in range(num_runs)]
    if method == 'linear_probe' and protocol_version == LP_GRID_PROTOCOL_VERSION:
        if run_seeds != [23, 24, 25, 26, 27]:
            raise ValueError(
                f"{LP_GRID_PROTOCOL_VERSION} requires exactly seeds 23-27; got {run_seeds}"
            )
        if not independent_validation_split:
            raise ValueError(
                f"{LP_GRID_PROTOCOL_VERSION} requires probe.independent_validation_split_per_seed=true"
            )
        if eval_split_group == 'fixed':
            raise ValueError(
                f"{LP_GRID_PROTOCOL_VERSION} cannot use a fixed validation split"
            )

    results = []
    validation_split_fingerprints = []
    for run_idx in range(num_runs):
        run_seed = base_seed + run_idx * seed_stride
        if eval_split_group == 'fixed':
            run_val_X = val_features
            run_val_y = val_labels
            run_test_X = test_features
            run_test_y = test_labels
            run_val_group_keys = val_group_keys
            run_test_group_keys = test_group_keys
            split_metadata = {
                'seed': int(run_seed),
                'strategy': 'fixed_manifest',
                'validation_size': int(len(run_val_y)),
                'test_size': int(len(run_test_y)),
                'validation_indices_sha256': 'fixed_manifest_validation',
                'test_indices_sha256': 'fixed_manifest_test',
            }
        elif eval_split_group == 'subject' or _uses_subject_eval_pooling(eval_pooling):
            val_idx, test_idx = _split_eval_pool_by_group(eval_group_keys, run_seed)
            val_idx_t = torch.from_numpy(val_idx).long()
            test_idx_t = torch.from_numpy(test_idx).long()
            run_val_group_keys = [eval_group_keys[idx] for idx in val_idx]
            run_test_group_keys = [eval_group_keys[idx] for idx in test_idx]
            run_val_X = eval_features[val_idx_t]
            run_val_y = eval_labels[val_idx_t]
            run_test_X = eval_features[test_idx_t]
            run_test_y = eval_labels[test_idx_t]
            split_metadata = {
                'seed': int(run_seed),
                'strategy': 'subject_resampled',
                'validation_size': int(len(val_idx)),
                'test_size': int(len(test_idx)),
                'validation_group_count': int(len(set(run_val_group_keys))),
                'test_group_count': int(len(set(run_test_group_keys))),
                'validation_indices_sha256': hashlib.sha256(
                    np.asarray(val_idx, dtype=np.int64).tobytes()
                ).hexdigest(),
                'test_indices_sha256': hashlib.sha256(
                    np.asarray(test_idx, dtype=np.int64).tobytes()
                ).hexdigest(),
            }
        else:
            val_idx, test_idx = _split_eval_pool(len(eval_labels), run_seed)
            val_idx_t = torch.from_numpy(val_idx).long()
            test_idx_t = torch.from_numpy(test_idx).long()
            run_val_group_keys = [eval_group_keys[idx] for idx in val_idx]
            run_test_group_keys = [eval_group_keys[idx] for idx in test_idx]
            run_val_X = eval_features[val_idx_t]
            run_val_y = eval_labels[val_idx_t]
            run_test_X = eval_features[test_idx_t]
            run_test_y = eval_labels[test_idx_t]
            split_metadata = {
                'seed': int(run_seed),
                'strategy': 'sample_resampled',
                'validation_size': int(len(val_idx)),
                'test_size': int(len(test_idx)),
                'validation_indices_sha256': hashlib.sha256(
                    np.asarray(val_idx, dtype=np.int64).tobytes()
                ).hexdigest(),
                'test_indices_sha256': hashlib.sha256(
                    np.asarray(test_idx, dtype=np.int64).tobytes()
                ).hexdigest(),
            }
        validation_split_fingerprints.append(split_metadata['validation_indices_sha256'])

        best_epoch = 0
        extra_checkpoint = None
        linear_probe_selection = None
        if method == 'mlp_probe':
            val_stats, test_stats, best_epoch, extra_checkpoint = _fit_and_evaluate_mlp_probe(
                task_type=task_type,
                num_classes=config['task']['num_classes'],
                probe_cfg=probe_cfg,
                seed=run_seed,
                train_X=train_X,
                train_y=train_y,
                val_X=run_val_X,
                val_y=run_val_y,
                test_X=run_test_X,
                test_y=run_test_y,
                device=device,
                rank=rank,
                val_group_keys=run_val_group_keys,
                test_group_keys=run_test_group_keys,
            )
        else:
            val_stats, test_stats, linear_probe_selection = _fit_and_evaluate_probe(
                method=method,
                task_type=task_type,
                probe_cfg=probe_cfg,
                seed=run_seed,
                train_X=train_X.numpy(),
                train_y=_to_numpy_labels(train_y, task_type),
                val_X=run_val_X.numpy(),
                val_y=_to_numpy_labels(run_val_y, task_type),
                test_X=run_test_X.numpy(),
                test_y=_to_numpy_labels(run_test_y, task_type),
                val_group_keys=run_val_group_keys,
                test_group_keys=run_test_group_keys,
            )
            if linear_probe_selection is not None:
                extra_checkpoint = {
                    'linear_probe_selection': linear_probe_selection,
                    'validation_split': split_metadata,
                }

        run_name = f"run_{run_idx:02d}_seed_{run_seed}"
        run_dir = output_dir / run_name
        run_config = _to_plain_dict(copy.deepcopy(config))
        run_config['experiment']['seed'] = run_seed
        run_config['experiment']['output_dir'] = str(run_dir)
        run_config.setdefault('probe', {})['validation_split'] = split_metadata
        if linear_probe_selection is not None:
            linear_cfg = run_config['probe'].setdefault('linear_probe', {})
            selected_key = f"selected_{linear_probe_selection['grid_parameter']}"
            linear_cfg[selected_key] = linear_probe_selection[selected_key]
            linear_cfg['grid_search_results'] = linear_probe_selection['grid_search_results']

        if rank == 0:
            logger.info(
                f"{run_name} - Val: "
                + " | ".join([f"{k}: {v:.4f}" for k, v in val_stats.items()])
            )
            logger.info(
                f"{run_name} - Test: "
                + " | ".join([f"{k}: {v:.4f}" for k, v in test_stats.items()])
            )

            _save_probe_run_artifact(
                run_dir,
                run_config,
                val_stats,
                test_stats,
                epoch=(best_epoch + 1) if method == 'mlp_probe' else best_epoch,
                extra_checkpoint=extra_checkpoint,
            )
            _log_wandb_metrics(
                {f"{run_name}/val_{k}": v for k, v in val_stats.items()} |
                {f"{run_name}/test_{k}": v for k, v in test_stats.items()},
                step=run_idx,
            )

        result = {
            'run_name': run_name,
            'seed': run_seed,
            'validation_split': split_metadata,
            'val_stats': val_stats,
            'test_stats': test_stats,
        }
        if linear_probe_selection is not None:
            result['linear_probe_selection'] = linear_probe_selection
            result[f"selected_{linear_probe_selection['grid_parameter']}"] = linear_probe_selection[
                f"selected_{linear_probe_selection['grid_parameter']}"
            ]
        results.append(result)

    if rank == 0:
        if independent_validation_split and len(set(validation_split_fingerprints)) != len(
            validation_split_fingerprints
        ):
            raise RuntimeError(
                "Per-seed validation splits are not independent: duplicate validation split fingerprint detected"
            )
        _write_probe_summary(output_dir, method, results, probe_cfg=probe_cfg)


def run_full_finetune(
    config,
    output_dir,
    checkpoint_dir,
    model,
    model_without_ddp,
    train_loader,
    val_loader,
    test_loader,
    train_sampler,
    rank,
    label_scaler,
):
    task_config = config['task']

    if task_config['task_type'] == 'classification':
        criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    else:
        criterion = nn.MSELoss()

    optimizer = create_downstream_optimizer(model_without_ddp, config)
    scheduler = create_downstream_scheduler(optimizer, config, len(train_loader))
    scaler = GradScaler() if config['training']['use_amp'] else None

    best_metric = 0.0
    best_loss = float('inf')
    best_epoch = -1
    start_epoch = 0

    if config['experiment'].get('resume', None) is not None:
        start_epoch, best_metric, best_loss = load_checkpoint(
            config['experiment']['resume'],
            model_without_ddp,
            optimizer,
            scheduler,
            scaler
        )
        logger.info(f"Resumed from epoch {start_epoch}. Best metric: {best_metric:.4f}, Best loss: {best_loss:.4f}")
    else:
        best_metric = -best_loss

    if rank == 0:
        logger.info("Starting fine-tuning...")
        logger.info(f"Training from epoch {start_epoch} to {config['optim']['epochs']}")

    for epoch in range(start_epoch, config['optim']['epochs']):
        if dist.is_available() and dist.is_initialized() and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_stats = finetune_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            epoch, config, rank, label_scaler
        )

        if rank == 0:
            logger.info(f"Epoch {epoch} Training - " + " | ".join([f"{k}: {v:.4f}" for k, v in train_stats.items()]))

        if epoch % config['validation']['val_freq'] == 0 or epoch == config['optim']['epochs'] - 1:
            val_stats = evaluate(model, val_loader, criterion, config, rank, epoch, label_scaler, 'val')
            test_stats = evaluate(model, test_loader, criterion, config, rank, epoch, label_scaler, 'test')

            if rank == 0:
                logger.info(f"Epoch {epoch} Validation - " + " | ".join([f"{k}: {v:.4f}" for k, v in val_stats.items()]))
                logger.info(f"Epoch {epoch} Test - " + " | ".join([f"{k}: {v:.4f}" for k, v in test_stats.items()]))

                is_best = val_stats['loss'] < best_loss
                if is_best:
                    best_loss = val_stats['loss']
                    best_metric = -best_loss
                    best_epoch = epoch

                eval_step = (epoch + 1) * len(train_loader)
                wandb_payload = {"epoch": epoch}
                wandb_payload.update({f"val/{k}": v for k, v in val_stats.items()})
                wandb_payload.update({f"test/{k}": v for k, v in test_stats.items()})
                wandb_payload["best/val_loss"] = best_loss
                wandb_payload["best/epoch"] = best_epoch
                _log_wandb_metrics(wandb_payload, step=eval_step)

                checkpoint_state = {
                    'epoch': epoch + 1,
                    'model_state_dict': model_without_ddp.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_metric': best_metric,
                    'best_loss': best_loss,
                    'config': _to_plain_dict(config),
                    'train_stats': train_stats,
                    'val_stats': val_stats,
                    'test_stats': test_stats,
                }
                if scaler is not None:
                    checkpoint_state['scaler_state_dict'] = scaler.state_dict()

                if is_best or (epoch + 1) % config['logging']['save_freq'] == 0:
                    save_checkpoint(
                        state_dict=checkpoint_state,
                        output_dir=checkpoint_dir,
                        epoch=epoch,
                        is_best=is_best,
                        rank=rank,
                        strategy='ddp',
                        filename=f'checkpoint_epoch_{epoch}.pth'
                    )
                    logger.info(f"Checkpoint saved at epoch {epoch}")

                if is_best:
                    logger.info(f"New best validation loss: {best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description='fMRI Downstream Fine-tuning')
    parser.add_argument('--config', type=str, default='configs/finetune_config.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (overrides config)')
    parser.add_argument('--skip_pretrained_bias', action='store_true', help='Skip all bias parameters when loading pretrained checkpoint')
    args = parser.parse_args()

    config = load_config(args.config)
    if args.resume is not None:
        config['experiment']['resume'] = args.resume
    if args.output_dir is not None:
        config['experiment']['output_dir'] = args.output_dir
    if args.skip_pretrained_bias:
        config['experiment']['skip_bias_in_pretrained'] = True

    mode = _get_mode(config)
    is_distributed, rank, world_size, gpu = setup_distributed()
    pretrained_checkpoint_path = config['experiment'].get('pretrained_checkpoint', None)
    use_fsdp_sharded_probe = (
        is_distributed
        and mode in {'linear_probe', 'ridge', 'svm', 'mlp_probe'}
        and _is_rank_shard_checkpoint(pretrained_checkpoint_path)
    )
    if use_fsdp_sharded_probe:
        config['experiment']['defer_pretrained_load'] = True

    set_seed(config['experiment']['seed'], rank)
    output_dir = Path(config['experiment']['output_dir'])

    if rank == 0:
        checkpoint_dir = output_dir / 'checkpoints'
        log_dir = output_dir / 'logs'
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        _save_runtime_config(config, output_dir)
        with open(output_dir / 'training_log.txt', 'w') as f:
            f.write(f"Fine-tuning started at {datetime.datetime.now()}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Config: {args.config}\n")
            f.write(f"Output directory: {config['experiment']['output_dir']}\n")
            f.write(f"Mode: {mode}\n")
            f.write(f"Task type: {config['task']['task_type']}\n")
    else:
        checkpoint_dir = None

    logger = setup_logger(output_dir, name="neurojepa", rank=rank)
    wandb_run = _init_wandb(config, output_dir, rank)

    try:
        if is_distributed:
            dist.barrier()

        if rank == 0:
            logger.info(f"Config: {args.config}")
            logger.info(f"Output directory: {config['experiment']['output_dir']}")
            logger.info(f"Mode: {mode}")
            logger.info(f"Task type: {config['task']['task_type']}")
            logger.info(f"Num classes: {config['task']['num_classes']}")
            logger.info(f"Distributed: {is_distributed}")

        device = _get_runtime_device(gpu)
        if rank == 0 and device.type != 'cuda':
            logger.info("CUDA is unavailable in the current environment; running probe extraction on CPU.")
        if use_fsdp_sharded_probe:
            if device.type != 'cuda':
                raise RuntimeError('FSDP rank-sharded downstream requires CUDA.')
            model = build_backbone(config, downstream=False).to(device)
            model = wrap_submodule_fsdp(model, reduce_fp32=True)
            used_path, incompat = _load_fsdp_rank_sharded_pretrained(
                model, pretrained_checkpoint_path, rank, prefix='backbone.'
            )
            if rank == 0:
                logger.info('Loaded FSDP rank-sharded pretrained checkpoint from %s', used_path)
                logger.info('FSDP shard load missing=%d unexpected=%d', len(incompat.missing_keys), len(incompat.unexpected_keys))
        else:
            model = create_model(config).to(device)

        if mode == 'full_finetune' and config['training'].get('freeze_encoder', True):
            if rank == 0:
                logger.info("Freezing encoder weights. Only the head will be trained.")
            trainable_keys = ('head', 'prototypes', 'prototype_logit_scale')
            for name, param in model.named_parameters():
                if not any(k in name for k in trainable_keys):
                    param.requires_grad = False
            if rank == 0:
                logger.info("Trainable parameters:")
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        logger.info(name)

        if mode == 'full_finetune' and is_distributed:
            model = DDP(model, device_ids=[gpu], find_unused_parameters=True)

        model_without_ddp = model if use_fsdp_sharded_probe else (model.module if hasattr(model, 'module') else model)

        train_loader, val_loader, test_loader, train_sampler = create_downstream_dataloaders(
            config, is_distributed, rank, world_size
        )

        label_scaler = None
        if config['task']['task_type'] == 'regression':
            mean_val, scale_val = _get_train_regression_label_stats(train_loader.dataset)
            config['task']['mean'] = mean_val
            config['task']['std'] = scale_val
            if rank == 0:
                logger.info(
                    f"Auto-computed regression label stats from training labels only. "
                    f"Mean: {mean_val:.4f}, Std: {scale_val:.4f}"
                )
                _save_runtime_config(config, output_dir)
            label_scaler = LabelScaler(
                torch.tensor(mean_val, device=gpu, dtype=torch.float32),
                torch.tensor(scale_val, device=gpu, dtype=torch.float32),
            )

        if rank == 0:
            logger.info(f"Training samples: {len(train_loader.dataset)}")
            logger.info(f"Validation samples: {len(val_loader.dataset)}")
            logger.info(f"Test samples: {len(test_loader.dataset)}")
            logger.info(f"Batches per epoch: {len(train_loader)}")

        if mode == 'full_finetune':
            run_full_finetune(
                config=config,
                output_dir=output_dir,
                checkpoint_dir=checkpoint_dir,
                model=model,
                model_without_ddp=model_without_ddp,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                train_sampler=train_sampler,
                rank=rank,
                label_scaler=label_scaler,
            )
        elif mode in {'linear_probe', 'ridge', 'svm', 'mlp_probe'}:
            run_feature_based_probe(
                config=config,
                output_dir=output_dir,
                model=model_without_ddp,
                train_dataset=train_loader.dataset,
                val_dataset=val_loader.dataset,
                test_dataset=test_loader.dataset,
                rank=rank,
                gpu=gpu,
            )
        else:
            raise ValueError(f"Unsupported experiment.mode: {mode}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        cleanup_distributed()


if __name__ == '__main__':
    main()
