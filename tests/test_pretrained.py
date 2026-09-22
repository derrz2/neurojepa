import pytest
import torch
from neurojepa import available_models, load_model


@pytest.mark.parametrize('variant', list(available_models()))
def test_released_checkpoint(variant):
    spec = available_models()[variant]
    if spec['status'] != 'available':
        with pytest.raises(FileNotFoundError, match='not been recovered'):
            load_model(variant)
        return
    model = load_model(variant)
    assert not model.training
    assert sum(p.numel() for p in model.parameters()) == spec['backbone_parameters']
    with torch.inference_mode():
        y = model(torch.randn(2, 100, 200))
    assert y.shape == (2, spec['model']['embed_dim'])
    assert torch.isfinite(y).all()


def test_unknown_variant():
    with pytest.raises(ValueError, match='Unknown variant'):
        load_model('not-a-model')


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match='Checkpoint not found'):
        load_model('10m', checkpoint_dir=tmp_path)


def test_checksum_rejects_modified_file(tmp_path):
    filename = available_models()['10m']['weights']['filename']
    (tmp_path / filename).write_bytes(b'not the released weights')
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        load_model('10m', checkpoint_dir=tmp_path)
