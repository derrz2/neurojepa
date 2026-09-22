from __future__ import annotations

import torch
import torch.nn as nn
import timm

from .base import GridBackboneBase, cfg_get


_CONVNEXT_MODEL_NAMES = {
    "convnext2d_tiny": "convnext_tiny",
    "convnext2d_small": "convnext_small",
    "convnext2d_base": "convnext_base",
    "convnext2d_large": "convnext_large",
    "convnext2d_xlarge": "convnext_xlarge",
}

_CONVNEXT_STAGE_DEPTHS = {
    "convnext2d_tiny": [3, 3, 9, 3],
    "convnext2d_small": [3, 3, 27, 3],
    "convnext2d_base": [3, 3, 27, 3],
    "convnext2d_large": [3, 3, 27, 3],
    "convnext2d_xlarge": [3, 3, 27, 3],
}


class ConvNeXt2DBackbone(GridBackboneBase):
    def __init__(self, args=None, *, variant: str = "convnext2d_base", downstream=None, num_classes=None) -> None:
        model_cfg = cfg_get(args, "model", args)
        feature_dim = int(cfg_get(model_cfg, "embed_dim", 768))
        drop_rate = float(cfg_get(model_cfg, "drop_rate", 0.0))
        downstream_flag = bool(cfg_get(model_cfg, "downstream", False) if downstream is None else downstream)
        n_classes = int(cfg_get(model_cfg, "num_classes", 1000) if num_classes is None else num_classes)
        super().__init__(
            cfg=args,
            feature_dim=feature_dim,
            downstream=downstream_flag,
            num_classes=n_classes,
            drop_rate=drop_rate,
        )

        if variant not in _CONVNEXT_MODEL_NAMES:
            raise ValueError(f"Unsupported ConvNeXt2D variant: {variant}")

        encoder = timm.create_model(
            _CONVNEXT_MODEL_NAMES[variant],
            pretrained=False,
            in_chans=1,
            num_classes=0,
            global_pool="avg",
        )

        self.encoder = encoder
        self.backbone_family = "convnext2d"
        self.variant = variant
        self.stage_depths = _CONVNEXT_STAGE_DEPTHS[variant]
        self.num_layers_for_decay = 5

        in_dim = int(encoder.num_features)
        self.proj = nn.Identity() if in_dim == self.feature_dim else nn.Linear(in_dim, self.feature_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(x)
        return self.encoder(x)

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

    def get_layer_id(self, param_name: str) -> int:
        name = param_name.replace("_fsdp_wrapped_module.", "")
        if name.startswith("backbone."):
            name = name[len("backbone.") :]

        if name.startswith("encoder.stem"):
            return 0
        if name.startswith("encoder.stages."):
            return int(name.split(".")[2]) + 1
        return self.num_layers_for_decay
