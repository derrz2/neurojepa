"""Train a fixed-C linear classifier on cached features; evaluate held-out data."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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
    for split in ['train', 'test']:
        parser.add_argument(f'--{split}-features', required=True)
        parser.add_argument(f'--{split}-labels', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--C', type=float, default=1.0, help='Fixed before test evaluation; tune only on a separate validation split.')
    args = parser.parse_args()
    if not np.isfinite(args.C) or args.C <= 0:
        parser.error('--C must be finite and positive')
    if args.output_dir.exists():
        parser.error('--output-dir already exists; choose a new directory')
    x_train, y_train = read_split(args.train_features, args.train_labels)
    x_test, y_test = read_split(args.test_features, args.test_labels)
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError('Train/test feature dimensions must match; use the same encoder.')
    if len(np.unique(y_train)) < 2 or not set(y_test).issubset(set(y_train)):
        raise ValueError('Training requires at least two classes and must cover all test classes.')
    # Fit BOTH feature standardization and the classifier on training rows only.
    probe = make_pipeline(StandardScaler(), LogisticRegression(C=args.C, max_iter=2000, random_state=42))
    probe.fit(x_train, y_train)
    classifier = probe[-1]
    if np.max(classifier.n_iter_) >= classifier.max_iter:
        raise RuntimeError('Classifier did not converge; adjust training settings without using test results.')
    prediction = probe.predict(x_test)
    report = {'protocol': 'fixed-C frozen-feature logistic regression; not the paper benchmark protocol',
              'train_samples': len(x_train), 'test_samples': len(x_test),
              'feature_dim': x_train.shape[1], 'C': args.C,
              'test_accuracy': float(accuracy_score(y_test, prediction)),
              'test_balanced_accuracy': float(balanced_accuracy_score(y_test, prediction)),
              'test_macro_f1': float(f1_score(y_test, prediction, average='macro', zero_division=0))}
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
