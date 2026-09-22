"""Validate imports, released encoders, and a synthetic optimizer step."""
import argparse
import importlib
import json
import platform
import sys
from pathlib import Path

import torch
from neurojepa import available_models, load_model

# Training entry points are source-checkout scripts, not installed packages.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--pretrain-smoke', action='store_true',
                        help='Also run one synthetic pretraining backward/optimizer step.')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA requested but unavailable: check the PyTorch build and NVIDIA driver.')
    for name in ['numpy', 'pandas', 'scipy', 'sklearn', 'nibabel', 'nilearn',
                 'timm', 'torchvision', 'torchmetrics', 'einops', 'loguru',
                 'fvcore', 'yaml', 'safetensors', 'pretrain', 'finetune']:
        importlib.import_module(name)
    report = {'python': platform.python_version(), 'torch': torch.__version__,
              'cuda_runtime': torch.version.cuda, 'device': args.device, 'models': {}}
    if args.device == 'cuda':
        report['gpu'] = torch.cuda.get_device_name(0)
    torch.manual_seed(42)
    for variant, spec in available_models().items():
        encoder = load_model(variant, device=args.device)
        encoder.requires_grad_(False)
        x = torch.randn(2, 100, 200, device=args.device)
        with torch.no_grad():
            features = encoder(x)
        assert features.shape == (2, spec['model']['embed_dim'])
        assert torch.isfinite(features).all()
        # Exercise autograd + optimizer on a downstream head without altering
        # pretrained weights or requiring any participant data.
        head = torch.nn.Linear(features.shape[-1], 2).to(args.device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
        before = head.weight.detach().clone()
        labels = torch.tensor([0, 1], device=args.device)
        loss = torch.nn.functional.cross_entropy(head(features), labels)
        optimizer.zero_grad()
        loss.backward()
        assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
        optimizer.step()
        assert not torch.equal(before, head.weight)
        report['models'][variant] = {'parameters': sum(p.numel() for p in encoder.parameters()),
                                     'features': list(features.shape), 'finite_loss': float(loss.detach()),
                                     'optimizer_step': 'passed'}
        del encoder, head, optimizer, features
    if args.pretrain_smoke:
        from src.models import build_pretrain_model
        from src.utils import build_optimizer, apply_scaling_rules_to_cfg
        from src.utils.logging_utils import load_config
        cfg = load_config(str(Path(__file__).resolve().parents[1] / 'configs/pretrain_example.yaml'))
        cfg = apply_scaling_rules_to_cfg(cfg)
        model = build_pretrain_model(cfg).to(args.device).train()
        optimizer = build_optimizer(cfg, model.get_params_groups())
        views = [torch.randn(2, 100, 200, device=args.device) for _ in range(2)]
        views += [torch.randn(2, 40, 80, device=args.device) for _ in range(6)]
        loss = model(views)[0]
        assert torch.isfinite(loss)
        optimizer.zero_grad()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        updated = next(p for p in model.parameters() if p.grad is not None and torch.count_nonzero(p.grad))
        before = updated.detach().clone()
        optimizer.step()
        assert not torch.equal(before, updated)
        report['synthetic_pretraining'] = {'finite_loss': float(loss.detach()),
                                          'optimizer_step': 'passed', 'views': 8}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
