"""Extract frozen embeddings from preprocessed (N, ROI, time) NumPy arrays."""
import argparse
from pathlib import Path
import numpy as np
import torch
from neurojepa import load_model

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--variant', default='10m', choices=['2m', '10m'])
parser.add_argument('--checkpoint-dir', default=None)
parser.add_argument('--device', default='cpu')
parser.add_argument('--input', type=Path, help='Floating-point .npy array with shape (N, ROI, time).')
parser.add_argument('--output', type=Path, help='Save ordered embeddings to a new .npy file.')
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--threads', type=int, default=2)
args = parser.parse_args()
if args.batch_size < 1 or args.threads < 1:
    parser.error('batch-size and threads must be positive')
if args.input is not None and args.output is None:
    parser.error('--input requires --output')
if args.output is not None:
    if args.output.suffix != '.npy':
        parser.error('--output must end with .npy')
    if args.output.exists():
        parser.error('--output already exists; choose a new path')
torch.set_num_threads(args.threads)
model = load_model(args.variant, checkpoint_dir=args.checkpoint_dir, device=args.device)
torch.manual_seed(0)
if args.input is None:
    print('Synthetic smoke test only; no study data or scientific metrics.')
    data = torch.randn(2, 100, 200).numpy()
else:
    data = np.load(args.input, mmap_mode='r', allow_pickle=False)
if data.ndim != 3 or min(data.shape) < 1 or not np.issubdtype(data.dtype, np.floating):
    raise ValueError('Expected a nonempty floating-point array shaped (N, ROI, time).')
chunks = []
for start in range(0, len(data), args.batch_size):
    x = torch.from_numpy(np.array(data[start:start + args.batch_size], dtype=np.float32, copy=True)).to(args.device)
    if not torch.isfinite(x).all():
        raise ValueError('Inputs contain NaN or infinity.')
    with torch.inference_mode():
        features = model(x)
    if not torch.isfinite(features).all():
        raise ValueError('Encoder produced nonfinite features.')
    chunks.append(features.cpu().numpy())
features = np.concatenate(chunks)
if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        np.save(stream, features, allow_pickle=False)
    print(f'saved={args.output}')
print(f'parameters={sum(p.numel() for p in model.parameters()):,}')
print(f'input={tuple(data.shape)} features={tuple(features.shape)}')
