from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def cfg_get(cfg_obj: Any, key: str, default=None):
    if cfg_obj is None:
        return default
    if isinstance(cfg_obj, dict):
        return cfg_obj.get(key, default)
    if hasattr(cfg_obj, "get"):
        return cfg_obj.get(key, default)
    return getattr(cfg_obj, key, default)


class GridBackboneBase(nn.Module):
    def __init__(
        self,
        *,
        cfg=None,
        feature_dim: int,
        downstream: bool = False,
        num_classes: int = 1000,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.feature_dim = int(feature_dim)
        self.embed_dim = int(feature_dim)
        self.downstream = bool(downstream)
        self.num_classes = int(num_classes)
        self.backbone_family = "grid"
        self.stage_depths = []
        self.num_layers_for_decay = 0

        self.head_drop = nn.Dropout(float(drop_rate))
        if self.downstream:
            self.head = nn.Linear(self.feature_dim, self.num_classes)
        else:
            self.head = nn.Identity()

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, C, T), got {tuple(x.shape)}")
        return x.unsqueeze(1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_embedding(x)
        x = self.head_drop(x)
        return self.head(x)

    def extract_probe_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.forward_embedding(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_embedding(x)
        if not self.downstream:
            return x
        x = self.head_drop(x)
        return self.head(x)

    def get_layer_id(self, param_name: str) -> int:
        return 0
