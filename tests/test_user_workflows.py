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
    for split, n in [('train', 20), ('test', 8)]:
        x = rng.normal(size=(n, 4)).astype('float32')
        y = np.arange(n) % 2
        x[:, 0] += y * 2
        np.save(tmp_path / f'{split}_x.npy', x)
        np.save(tmp_path / f'{split}_y.npy', y)
    out = tmp_path / 'probe'
    run_script('examples/linear_probe.py', '--train-features', tmp_path / 'train_x.npy',
               '--train-labels', tmp_path / 'train_y.npy', '--test-features', tmp_path / 'test_x.npy',
               '--test-labels', tmp_path / 'test_y.npy', '--output-dir', out)
    params = np.load(out / 'probe.npz', allow_pickle=False)
    train = np.load(tmp_path / 'train_x.npy')
    np.testing.assert_allclose(params['mean'], train.mean(0), rtol=1e-6, atol=1e-7)
    test = np.load(tmp_path / 'test_x.npy')
    score = ((test - params['mean']) / params['scale']) @ params['coef'].T + params['intercept']
    predicted = params['classes'][(score[:, 0] > 0).astype(int)]
    np.testing.assert_array_equal(predicted, np.load(out / 'test_predictions.npy'))
    report = json.loads((out / 'metrics.json').read_text())
    assert report['train_samples'] == 20 and report['test_samples'] == 8


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
