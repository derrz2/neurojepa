"""Strict, pickle-free loading of released encoder weights."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

_CONFIG_DIR = Path(__file__).parent / 'configs'


def available_models() -> dict:
    """Return architecture metadata, including explicit weight availability."""
    return {p.stem: json.loads(p.read_text(encoding='utf-8'))
            for p in sorted(_CONFIG_DIR.glob('*.json'))}


def load_model(variant: str = '10m', *, checkpoint_dir=None,
               device: str | torch.device = 'cpu') -> torch.nn.Module:
    """Load an evaluation-mode encoder; never fall back to random weights.

    Inputs are preprocessed floating-point tensors shaped (batch, ROI, time).
    Outputs are final-layer pooled embeddings, not diagnostic predictions.
    This function does not preprocess or normalize the input.
    """
    models = available_models()
    if variant not in models:
        raise ValueError(f'Unknown variant {variant!r}; choose from {list(models)}')
    spec = models[variant]
    if spec['status'] != 'available':
        raise FileNotFoundError(
            f'{variant}: the requested pretrained checkpoint has not been recovered. '
            'No alternative epoch or random initialization will be substituted.'
        )
    root = (Path(checkpoint_dir) if checkpoint_dir is not None
            else Path(__file__).resolve().parents[1] / 'checkpoints')
    path = root / spec['weights']['filename']
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {path}. Pass checkpoint_dir explicitly.')
    digest = _sha256(path)
    if digest != spec['weights']['sha256']:
        raise ValueError('Checkpoint SHA-256 mismatch; obtain the matching release weights.')
    from src.models.backbone import build_backbone
    model = build_backbone({'model': spec['model']}, downstream=False)
    state = load_file(str(path), device='cpu')
    model.load_state_dict(state, strict=True)
    count = sum(p.numel() for p in model.parameters())
    if count != spec['backbone_parameters']:
        raise ValueError(f'Parameter count mismatch: {count} != {spec["backbone_parameters"]}')
    model.eval()
    return model.to(device)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
