import yaml
from pathlib import Path
from typing import Any

import logging
import sys
import os    

class ConfigBox(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Recursively convert nested dictionaries into ConfigBoxes
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = ConfigBox(value)

    def __getattr__(self, key):
        """Allows accessing dictionary keys as attributes."""
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        """Allows setting dictionary keys as attributes."""
        # If the new value is a dictionary, convert it to a ConfigBox too
        if isinstance(value, dict):
            self[key] = ConfigBox(value)
        else:
            self[key] = value

    def __delattr__(self, key):
        """Allows deleting dictionary keys as attributes."""
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")


class IgnoreUnknownLoader(yaml.SafeLoader):
    """YAML loader that degrades unknown Python tags into plain containers."""


def _unknown_constructor(loader, tag_suffix, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


IgnoreUnknownLoader.add_multi_constructor("", _unknown_constructor)


def _deep_update(base, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def unwrap_configbox(value: Any):
    if isinstance(value, dict):
        if set(value.keys()) == {"dictitems"}:
            return unwrap_configbox(value["dictitems"])
        return {key: unwrap_configbox(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [unwrap_configbox(item) for item in value]
    return value


def _to_plain_data(value: Any):
    if isinstance(value, ConfigBox):
        return {key: _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {key: _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_data(item) for item in value]
    return value


def _load_yaml_config(config_path: Path) -> dict:
    text = config_path.read_text(encoding="utf-8")
    try:
        raw = yaml.safe_load(text)
    except yaml.constructor.ConstructorError:
        raw = yaml.load(text, Loader=IgnoreUnknownLoader)

    if raw is None:
        return {}

    raw = unwrap_configbox(raw)
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported config type from {config_path}: {type(raw).__name__}")
    return raw


def _load_checkpoint_config(config_path: Path) -> dict:
    import torch

    checkpoint_obj = torch.load(config_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint_obj, dict):
        raise TypeError(f"Unsupported checkpoint type from {config_path}: {type(checkpoint_obj).__name__}")
    if "config" not in checkpoint_obj:
        raise KeyError(f"Checkpoint does not contain a 'config' entry: {config_path}")

    raw = unwrap_configbox(checkpoint_obj["config"])
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported checkpoint config type from {config_path}: {type(raw).__name__}")
    return raw


def _merge_backbone_config(config_dict: dict, config_path: Path) -> dict:
    config_dict = dict(config_dict)
    repo_root = Path(__file__).resolve().parents[2]

    backbone_cfg_path = config_dict.get("backbone_config") or config_dict.get("vit_config")
    if backbone_cfg_path:
        backbone_candidates = [
            Path(backbone_cfg_path),
            config_path.parent / str(backbone_cfg_path),
            repo_root / str(backbone_cfg_path),
            repo_root / "configs" / str(backbone_cfg_path),
        ]
        backbone_path = next((p for p in backbone_candidates if p.exists()), None)
        if backbone_path is None:
            raise FileNotFoundError(f"Backbone config not found for {config_path}: {backbone_cfg_path}")
        with open(backbone_path, "r", encoding="utf-8") as f:
            backbone_dict = yaml.safe_load(f) or {}
        vit_model = backbone_dict.get("model", backbone_dict)
        if "model" not in config_dict or config_dict["model"] is None:
            config_dict["model"] = {}
        config_dict["model"] = _deep_update(vit_model, config_dict["model"])
    return config_dict


def load_config(config_path):
    config_path = Path(config_path)
    suffix = config_path.suffix.lower()

    if suffix in {".pth", ".pt", ".ckpt"}:
        config_dict = _load_checkpoint_config(config_path)
    else:
        config_dict = _load_yaml_config(config_path)

    config_dict = _merge_backbone_config(config_dict, config_path)
    return ConfigBox(config_dict)


def save_config(config, config_path):
    config_path = Path(config_path)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(_to_plain_data(config), f, default_flow_style=False, sort_keys=False)

def setup_logger(output_dir, name="neurojepa", rank=0):

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  

    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    if rank == 0:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if rank == 0 and output_dir is not None:
        log_file = os.path.join(output_dir, "training_log.txt")
        fh = logging.FileHandler(log_file, mode='a') 
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
