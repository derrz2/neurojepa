from copy import deepcopy

import torch
import pytest
from safetensors.torch import load_file, save_file
from neurojepa import available_models, load_model
from src.models import create_model
from src.models.backbone import build_backbone
from src.utils.checkpoint import load_checkpoint


@pytest.mark.parametrize('variant', ['2m', '10m'])
def test_downstream_encoder_matches_pretrained(variant):
    spec = available_models()[variant]
    cfg = {
        'model': deepcopy(spec['model']),
        'task': {'num_classes': 2},
        'experiment': {'pretrained_checkpoint': 'checkpoints/' + spec['weights']['filename']},
    }
    model = create_model(cfg).eval()
    encoder = load_model(variant)
    torch.manual_seed(42)
    x = torch.randn(2, 100, 200)
    with torch.inference_mode():
        expected = encoder(x)
        actual = model(x, return_probe_features=True)
        logits = model(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert logits.shape == (2, 2)


@pytest.mark.parametrize('fault', ['missing_key', 'wrong_shape'])
@pytest.mark.parametrize('variant', ['2m', '10m'])
def test_downstream_rejects_incompatible_encoder(tmp_path, fault, variant):
    spec = available_models()[variant]
    model = build_backbone({'model': deepcopy(spec['model'])}, downstream=True, num_classes=2)
    state = load_file('checkpoints/' + spec['weights']['filename'])
    if fault == 'missing_key':
        del state['cls_token']
    else:
        state['cls_token'] = state['cls_token'][..., :1].contiguous()
    path = tmp_path / 'incompatible.safetensors'
    save_file(state, str(path))
    with pytest.raises(RuntimeError):
        load_checkpoint(str(path), model, mode='pretrained', target='backbone')


@pytest.mark.parametrize('variant', ['2m', '10m'])
def test_downstream_rejects_modified_named_release(tmp_path, variant):
    spec = available_models()[variant]
    path = tmp_path / spec['weights']['filename']
    path.write_bytes(b'corrupted')
    model = build_backbone({'model': deepcopy(spec['model'])}, downstream=True, num_classes=2)
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        load_checkpoint(str(path), model, mode='pretrained', target='backbone')
