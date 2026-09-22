"""Create a task-manifest downstream configuration matching a released encoder."""
import argparse
from pathlib import Path

import yaml
from neurojepa import available_models

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--variant', choices=['2m', '10m'], required=True)
parser.add_argument('--mode', choices=['linear_probe', 'full_finetune', 'frozen_head'], default='linear_probe')
parser.add_argument('--num-classes', type=int, default=2)
for split in ['train', 'val', 'test']:
    parser.add_argument(f'--{split}-manifest', default=f'data/{split}.csv')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists() or args.num_classes < 2:
    parser.error('Choose a new output path and num-classes >= 2.')
root = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((root / 'configs/finetune_example.yaml').read_text())
spec = available_models()[args.variant]
cfg['model'] = spec['model'] | {'downstream': True, 'fusion_mode': 'none'}
cfg['experiment']['pretrained_checkpoint'] = 'checkpoints/' + spec['weights']['filename']
cfg['experiment']['mode'] = 'full_finetune' if args.mode == 'frozen_head' else args.mode
cfg['task']['num_classes'] = args.num_classes
cfg['training']['freeze_encoder'] = args.mode != 'full_finetune'
cfg['validation']['save_best'] = True
cfg['data'].update(mode='task_manifest', raw_series_layout='channel_time', num_workers=0)
for split in ['train', 'val', 'test']:
    cfg['data'][split + '_list'] = [getattr(args, split + '_manifest')]
cfg['probe'] = {'protocol_version': 'user_example_fixed_split', 'num_runs': 1,
                'eval_split_group': 'fixed', 'eval_pooling': 'none',
                # Omit C_grid to use finetune.py's original default search grid.
                'linear_probe': {'max_iter': 2000}}
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open('x', encoding='utf-8') as stream:
    yaml.safe_dump(cfg, stream, sort_keys=False)
print(f'Saved {args.output}; inspect the task settings and supply subject-disjoint manifests before running.')
