"""Read configuration values from dictionaries or attribute-based configs."""
from typing import Any


def cfg_get(cfg_obj: Any, key: str, default=None):
    if cfg_obj is None:
        return default
    if isinstance(cfg_obj, dict):
        return cfg_obj.get(key, default)
    if hasattr(cfg_obj, "get"):
        return cfg_obj.get(key, default)
    return getattr(cfg_obj, key, default)
