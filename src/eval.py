import torch
import torch.nn.functional as F
import torch.distributed as dist
import contextlib
from src.utils.helpers import MetricLogger
from torch.cuda.amp import autocast
from torchmetrics import MetricCollection, Accuracy, F1Score, MeanSquaredError, MeanAbsoluteError, R2Score, PearsonCorrCoef
import logging

from src.utils.utils import forward_pass, move_to_device
logger = logging.getLogger("neurojepa")


def _compute_rankme(features: torch.Tensor) -> float:
    singular_values = torch.linalg.svdvals(features.float())
    singular_values = singular_values[singular_values > 0]
    if singular_values.numel() == 0:
        return 0.0

    probs = singular_values / singular_values.sum().clamp_min(1e-12)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
    return float(torch.exp(entropy).item())


def _get_rankme_n_global_views(config) -> int:
    hard_val_cfg = getattr(config.validation, "deterministic_multiview", None)
    if hard_val_cfg is not None:
        return int(getattr(hard_val_cfg, "n_global_views", 2))

    model_cfg = getattr(config.model, str(config.model_chose), None)
    if model_cfg is not None:
        return int(getattr(model_cfg, "n_global_views", 1))
    return 1


def _unpack_rankme_view(view):
    if isinstance(view, dict):
        return view["x"], view.get("patch_valid_mask")
    return view, None


@torch.no_grad()
def validate(model, val_loader, epoch, rank, config, teacher_temp=None):
    """Validate the model"""
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = f'Validation Epoch: [{epoch}]'
    device = torch.device(f'cuda:{rank}')

    for samples in metric_logger.log_every(val_loader, 50, header):
        # Move data to GPU
        if torch.cuda.is_available():
            samples = move_to_device(samples, device, non_blocking=True)
                
        # Forward pass
        with autocast(enabled=True):
            loss, outputs = forward_pass(model, samples, config, teacher_temp)

        metric_logger.update(loss=loss.item())
        for k, v in outputs.items():
            if k == 'collapse_metrics' and isinstance(v, dict):
                for ck, cv in v.items():
                    val = cv.item() if torch.is_tensor(cv) else cv
                    metric_logger.update(**{ck: val})
            elif torch.is_tensor(v):
                if v.ndim == 0:
                    metric_logger.update(**{k: v.item()})
            elif isinstance(v, (int, float)):
                metric_logger.update(**{k: v})

    metric_logger.synchronize_between_processes()
    if rank == 0:
        logger.info(f"Validation averaged stats: {metric_logger}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_rankme(model, val_loader, epoch, rank, config):
    if bool(getattr(config.training, "probe_val", False)):
        return {}

    rankme_cfg = getattr(config.validation, "rankme", None)
    if rankme_cfg is not None and not bool(getattr(rankme_cfg, "enable", False)):
        return {}

    model_ref = model.module if hasattr(model, "module") else model
    backbone = getattr(model_ref, "backbone", None)
    if backbone is None:
        return {}

    max_samples = int(getattr(rankme_cfg, "max_samples", 4096)) if rankme_cfg is not None else 4096
    if max_samples <= 0:
        return {}

    backbone.eval()
    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
    n_global_views = _get_rankme_n_global_views(config)

    collected = 0
    feature_chunks = []

    for samples in val_loader:
        if isinstance(samples, (list, tuple)):
            global_views = list(samples[:n_global_views])
        else:
            global_views = [samples]

        if len(global_views) == 0:
            continue

        global_views = [move_to_device(view, device, non_blocking=True) for view in global_views]
        global_tensors = []
        global_masks = []
        for view in global_views:
            x, patch_valid_mask = _unpack_rankme_view(view)
            global_tensors.append(x)
            global_masks.append(patch_valid_mask)
        global_batch = torch.cat(global_tensors, dim=0)
        key_padding_mask = None
        if any(mask is not None for mask in global_masks):
            if any(mask is None for mask in global_masks):
                raise ValueError("Mixed masked and unmasked views are not supported in RankMe validation.")
            key_padding_mask = torch.cat([~mask for mask in global_masks], dim=0)

        with autocast(enabled=torch.cuda.is_available()):
            features = backbone(global_batch, key_padding_mask=key_padding_mask)
        features = features.detach().float().contiguous()

        if dist.is_available() and dist.is_initialized():
            gathered = [torch.zeros_like(features) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, features)
            features = torch.cat(gathered, dim=0)

        batch_size = int(features.shape[0])
        take = min(batch_size, max_samples - collected)
        if rank == 0 and take > 0:
            feature_chunks.append(features[:take].cpu())
        collected += take

        if collected >= max_samples:
            break

    if rank != 0 or not feature_chunks:
        return {}

    feature_matrix = torch.cat(feature_chunks, dim=0)
    rankme_value = _compute_rankme(feature_matrix)
    logger.info(
        f"Validation RankMe Epoch [{epoch}] - "
        f"rankme_global_feat: {rankme_value:.4f} | samples: {feature_matrix.shape[0]} | dim: {feature_matrix.shape[1]}"
    )
    return {"rankme_global_feat": rankme_value}


@torch.no_grad()
def validate_probe_acc(model, val_loader, epoch, rank, config, teacher_temp=None):
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = f'Validation Epoch: [{epoch}]'
    m = model.module if hasattr(model, 'module') else model

    device = torch.device(f'cuda:{rank}')
    correct = torch.zeros(1, device=device)
    total = torch.zeros(1, device=device)
    loss_sum = torch.zeros(1, device=device)

    for samples, labels in metric_logger.log_every(val_loader, 50, header):
        samples = samples.cuda(rank, non_blocking=True)
        labels = labels.cuda(rank, non_blocking=True).long()

        with autocast(enabled=True):
            if hasattr(m, "extract_probe_features"):
                samples = m.extract_probe_features(samples)
            else:
                samples = m.backbone(samples)
        probe_param = next(m.probe.parameters())
        probe_input = samples.detach().to(device=probe_param.device, dtype=probe_param.dtype)
        autocast_ctx = (
            torch.autocast(device_type=probe_input.device.type, enabled=False)
            if probe_input.device.type != "cpu"
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            logits = m.probe(probe_input)
            loss = F.cross_entropy(logits.float(), labels)

        correct += (logits.argmax(1) == labels).sum()
        total += labels.numel()
        loss_sum += loss * labels.numel()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(correct)
        dist.all_reduce(total)
        dist.all_reduce(loss_sum)

    total = total.clamp_min(1.0)
    stats = {
        'loss': (loss_sum / total).item(),
        'acc': (correct / total).item(),
    }
    metric_logger.update(**stats)

    if rank == 0:
        logger.info(f"Validation averaged stats: {metric_logger}")

    return stats


@torch.no_grad()
def evaluate(model, data_loader, criterion, config, rank, epoch=None, label_scaler=None, mode='val'):

    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = f'{mode.capitalize()} Epoch: [{epoch}]' if epoch is not None else f'{mode.capitalize()}:'
    
    task_type = config['task']['task_type']
    device = torch.device(f'cuda:{rank}')

    metrics = None
    if task_type == 'classification':
        num_classes = config['task']['num_classes']
        metrics = MetricCollection({
            'acc': Accuracy(task="multiclass", num_classes=num_classes),
            'f1': F1Score(task="multiclass", num_classes=num_classes, average='weighted')
        })
    elif task_type == 'regression':
        metrics = MetricCollection({
            'mse': MeanSquaredError(),
            'mae': MeanAbsoluteError(),
            'r2': R2Score(),
            'corr': PearsonCorrCoef()
        })
    if metrics is not None:
        metrics.to(device)

    for batch in metric_logger.log_every(data_loader, 50, header, enable_log=False):

        samples, labels = batch
        loss = 0.0
        samples = samples.cuda(rank, non_blocking=True)
        labels = labels.cuda(rank, non_blocking=True)

        with torch.no_grad():
            model_outputs = model(samples)
            if isinstance(model_outputs, tuple):
                outputs = model_outputs[0]
            else:
                outputs = model_outputs

            if task_type == 'classification':
                labels = labels.squeeze().long() if labels.dim() > 1 else labels.long()
                task_loss = criterion(outputs, labels)
                metrics.update(outputs, labels)
            elif task_type == 'regression':
                target_norm = label_scaler.transform(labels) if label_scaler else labels
                task_loss = criterion(outputs.view_as(target_norm), target_norm)
                metrics.update(outputs.view(-1), target_norm.view(-1))
            else:
                raise ValueError(f"Unsupported task_type: {task_type}")

            loss = task_loss
            metric_logger.update(loss=loss.item())

    if metrics is not None:
        total_metrics = metrics.compute()
        for name, value in total_metrics.items():
            metric_logger.update(**{name: value.item()})

    metric_logger.synchronize_between_processes()
    
    if metrics is not None:
        metrics.reset()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
