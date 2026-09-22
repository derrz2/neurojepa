"""Smoke test: run from the repository root after `pip install -e .`."""
import argparse
import torch
from neurojepa import load_model

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--variant', default='10m', choices=['2m', '10m'])
parser.add_argument('--checkpoint-dir', default=None)
parser.add_argument('--device', default='cpu')
args = parser.parse_args()
model = load_model(args.variant, checkpoint_dir=args.checkpoint_dir, device=args.device)
torch.manual_seed(0)
# Synthetic inputs only. Real data require your study's preprocessing protocol.
x = torch.randn(2, 100, 200, device=args.device)
with torch.inference_mode():
    features = model(x)
assert torch.isfinite(features).all()
print(f'parameters={sum(p.numel() for p in model.parameters()):,}')
print(f'input={tuple(x.shape)} features={tuple(features.shape)}')
