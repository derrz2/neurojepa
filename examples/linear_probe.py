"""Use the original downstream runner: train, select C on validation, then test."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Use the source-checkout training entry point, not a separate probe implementation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finetune import _fit_and_evaluate_probe


def read_split(features, labels):
    x = np.load(features, allow_pickle=False)
    y = np.load(labels, allow_pickle=False)
    if x.ndim != 2 or not len(x) or not np.issubdtype(x.dtype, np.floating) or not np.isfinite(x).all():
        raise ValueError('Features must be a nonempty finite floating-point (N, D) array.')
    if y.shape != (len(x),) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError('Labels must be an integer (N,) array aligned with feature rows.')
    return x, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ['train', 'val', 'test']:
        parser.add_argument(f'--{split}-features', required=True)
        parser.add_argument(f'--{split}-labels', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--C-grid', type=float, nargs='+', default=None,
                        help='Candidate C values; defaults to finetune.py: 1e-5 through 1e3.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('--output-dir already exists; choose a new directory')
    x_train, y_train = read_split(args.train_features, args.train_labels)
    x_val, y_val = read_split(args.val_features, args.val_labels)
    x_test, y_test = read_split(args.test_features, args.test_labels)
    if len({x_train.shape[1], x_val.shape[1], x_test.shape[1]}) != 1:
        raise ValueError('Train/val/test feature dimensions must match; use the same encoder.')
    classes = np.unique(y_train)
    if len(classes) < 2 or not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError('Training labels must cover contiguous classes 0 .. K-1, with K >= 2.')
    if not set(y_val).union(y_test).issubset(set(classes)):
        raise ValueError('Training must cover every validation and test class.')
    probe_cfg = {'eval_pooling': 'none', 'linear_probe': {'max_iter': 2000}}
    if args.C_grid is not None:
        probe_cfg['linear_probe']['C_grid'] = args.C_grid
    val_stats, test_stats, selection, probe, test_outputs = _fit_and_evaluate_probe(
        'linear_probe', 'classification', probe_cfg, args.seed,
        x_train, y_train, x_val, y_val, x_test, y_test, return_estimator=True)
    classifier = probe[-1]
    scores = test_outputs.numpy()
    # Match the runner's >= 0 binary tie rule (sklearn.predict uses > 0).
    prediction = (scores.reshape(-1) >= 0).astype(np.int64) if scores.ndim == 1 or scores.shape[1] == 1 else scores.argmax(1)
    report = {'protocol': 'finetune.py validation-selected linear probe; supplied fixed splits',
              'train_samples': len(x_train), 'val_samples': len(x_val), 'test_samples': len(x_test),
              'feature_dim': x_train.shape[1], 'selected_C': selection['selected_C'],
              'val_stats': val_stats, 'test_stats': test_stats, 'selection': selection}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.save(args.output_dir / 'test_predictions.npy', prediction, allow_pickle=False)
    np.save(args.output_dir / 'test_probabilities.npy', probe.predict_proba(x_test), allow_pickle=False)
    np.savez(args.output_dir / 'probe.npz', classes=classifier.classes_,
             coef=classifier.coef_, intercept=classifier.intercept_,
             mean=probe[0].mean_, scale=probe[0].scale_)
    (args.output_dir / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
