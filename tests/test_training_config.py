"""Initialization only: does not run training or require participant data."""
import pytest


def test_pretrain_config_initializes_optimizer():
    pytest.importorskip('loguru')
    pytest.importorskip('scipy')
    from src.models import build_pretrain_model
    from src.utils import build_optimizer, apply_scaling_rules_to_cfg, DINOSchedulerWrapper
    from src.utils.logging_utils import load_config
    cfg = load_config('configs/pretrain_example.yaml')
    cfg = apply_scaling_rules_to_cfg(cfg)
    model = build_pretrain_model(cfg)
    optimizer = build_optimizer(cfg, model.get_params_groups())
    scheduler = DINOSchedulerWrapper(cfg, optimizer)
    assert optimizer.param_groups
    assert scheduler.current_step == 0
