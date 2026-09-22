"""Exercise the public file-based feature and downstream examples."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from neurojepa import load_model, available_models

ROOT = Path(__file__).resolve().parents[1]


def run_script(script, *args):
    return subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                          cwd=ROOT, check=True, capture_output=True, text=True)


@pytest.mark.parametrize('variant,dim', [('2m', 128), ('10m', 256)])
def test_extract_real_array_preserves_order(tmp_path, variant, dim):
    x = np.random.default_rng(42).standard_normal((3, 100, 200)).astype('float32')
    source, target = tmp_path / 'x.npy', tmp_path / 'features.npy'
    np.save(source, x)
    run_script('examples/extract_features.py', '--variant', variant, '--input', source,
               '--output', target, '--batch-size', 2, '--device', 'cpu')
    actual = np.load(target, allow_pickle=False)
    with torch.inference_mode():
        expected = load_model(variant)(torch.from_numpy(x)).numpy()
    assert actual.shape == (3, dim)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    with pytest.raises(subprocess.CalledProcessError):
        run_script('examples/extract_features.py', '--output', target)


def test_linear_probe_saved_outputs(tmp_path):
    pytest.importorskip('sklearn')
    rng = np.random.default_rng(42)
    for split, n in [('train', 20), ('val', 10), ('test', 8)]:
        x = rng.normal(size=(n, 4)).astype('float32')
        y = np.arange(n) % 2
        x[:, 0] += y * 2
        np.save(tmp_path / f'{split}_x.npy', x)
        np.save(tmp_path / f'{split}_y.npy', y)
    out = tmp_path / 'probe'
    run_script('examples/linear_probe.py', '--train-features', tmp_path / 'train_x.npy',
               '--train-labels', tmp_path / 'train_y.npy',
               '--val-features', tmp_path / 'val_x.npy', '--val-labels', tmp_path / 'val_y.npy',
               '--test-features', tmp_path / 'test_x.npy',
               '--test-labels', tmp_path / 'test_y.npy', '--output-dir', out)
    params = np.load(out / 'probe.npz', allow_pickle=False)
    train = np.load(tmp_path / 'train_x.npy')
    np.testing.assert_allclose(params['mean'], train.mean(0), rtol=1e-6, atol=1e-7)
    test = np.load(tmp_path / 'test_x.npy')
    score = ((test - params['mean']) / params['scale']) @ params['coef'].T + params['intercept']
    predicted = params['classes'][(score[:, 0] >= 0).astype(int)]
    np.testing.assert_array_equal(predicted, np.load(out / 'test_predictions.npy'))
    report = json.loads((out / 'metrics.json').read_text())
    assert report['train_samples'] == 20 and report['val_samples'] == 10 and report['test_samples'] == 8
    from finetune import _fit_and_evaluate_probe, DEFAULT_CLASSIFICATION_C_GRID
    val_stats, test_stats, selection = _fit_and_evaluate_probe(
        'linear_probe', 'classification', {'linear_probe': {'max_iter': 2000}}, 42,
        train, np.load(tmp_path / 'train_y.npy'), np.load(tmp_path / 'val_x.npy'),
        np.load(tmp_path / 'val_y.npy'), test, np.load(tmp_path / 'test_y.npy'))
    assert report['val_stats'] == val_stats and report['test_stats'] == test_stats
    assert report['selection'] == selection
    assert selection['grid'] == DEFAULT_CLASSIFICATION_C_GRID


@pytest.mark.parametrize('variant', ['2m', '10m'])
def test_generated_config_matches_release(tmp_path, variant):
    target = tmp_path / 'config.yaml'
    run_script('scripts/make_downstream_config.py', '--variant', variant, '--output', target)
    cfg = yaml.safe_load(target.read_text())
    spec = available_models()[variant]
    for name, value in spec['model'].items():
        assert cfg['model'][name] == value
    assert cfg['experiment']['pretrained_checkpoint'].endswith(spec['weights']['filename'])
    assert cfg['probe']['eval_split_group'] == 'fixed'
    assert cfg['data']['mode'] == 'task_manifest'
    assert 'C_grid' not in cfg['probe']['linear_probe']  # use runner default


@pytest.mark.parametrize('val_labels,expected_c', [([0, 1], 0.01), ([1, 0], 1.0)])
def test_validation_selects_before_test_and_test_labels_cannot_change_selection(monkeypatch, val_labels, expected_c):
    import finetune as runner
    train_x, val_x, test_x = np.ones((4, 2)), np.ones((2, 2)), np.zeros((2, 2))
    train_y = np.array([0, 1, 0, 1])
    fits, test_calls = [], []

    class Candidate:
        def __init__(self, c):
            self.c = c

        def fit(self, x, y):
            assert x is train_x and y is train_y  # never fit or refit on val/test
            fits.append(self.c)

    monkeypatch.setattr(runner, '_build_probe_estimator',
                        lambda *args, selected_hyperparameter: Candidate(selected_hyperparameter))

    def predict(model, task_type, x):
        if x is val_x:
            return torch.tensor([-1.0, 1.0]) if model.c == 0.01 else torch.tensor([1.0, -1.0])
        assert x is test_x and fits == [0.01, 1.0]
        test_calls.append(model.c)
        return torch.tensor([-0.5, 0.5])

    monkeypatch.setattr(runner, '_predict_probe_outputs', predict)
    for test_labels in ([0, 1], [1, 0]):
        fits.clear()
        _, _, selection = runner._fit_and_evaluate_probe(
            'linear_probe', 'classification', {'linear_probe': {'C_grid': [0.01, 1.0]}}, 42,
            train_x, train_y, val_x, np.array(val_labels), test_x, np.array(test_labels))
        assert selection['selected_C'] == expected_c
    assert test_calls == [expected_c, expected_c]  # only the selected model touches test


def test_cached_probe_requires_validation_split(tmp_path):
    with pytest.raises(subprocess.CalledProcessError) as exc:
        run_script('examples/linear_probe.py', '--train-features', 'unused.npy',
                   '--train-labels', 'unused.npy', '--test-features', 'unused.npy',
                   '--test-labels', 'unused.npy', '--output-dir', tmp_path / 'out')
    assert '--val-features' in exc.value.stderr and '--val-labels' in exc.value.stderr
