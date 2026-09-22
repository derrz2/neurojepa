from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

from .base import GridBackboneBase, cfg_get


_RESNET_BUILDERS: Dict[str, callable] = {
    "resnet2d_18": resnet18,
    "resnet2d_34": resnet34,
    "resnet2d_50": resnet50,
    "resnet2d_101": resnet101,
    "resnet2d_152": resnet152,
}

_RESNET_STAGE_DEPTHS: Dict[str, list[int]] = {
    "resnet2d_18": [2, 2, 2, 2],
    "resnet2d_34": [3, 4, 6, 3],
    "resnet2d_50": [3, 4, 6, 3],
    "resnet2d_101": [3, 4, 23, 3],
    "resnet2d_152": [3, 8, 36, 3],
}


class ResNet2DBackbone(GridBackboneBase):
    def __init__(self, args=None, *, variant: str = "resnet2d_18", downstream=None, num_classes=None) -> None:
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

        if variant not in _RESNET_BUILDERS:
            raise ValueError(f"Unsupported ResNet2D variant: {variant}")

        encoder = _RESNET_BUILDERS[variant](weights=None)
        stem_kernel = int(cfg_get(model_cfg, "stem_kernel", 7))
        stem_stride = int(cfg_get(model_cfg, "stem_stride", 2))
        stem_padding = stem_kernel // 2
        encoder.conv1 = nn.Conv2d(
            1,
            encoder.conv1.out_channels,
            kernel_size=stem_kernel,
            stride=stem_stride,
            padding=stem_padding,
            bias=False,
        )
        if not bool(cfg_get(model_cfg, "use_maxpool", True)):
            encoder.maxpool = nn.Identity()
        encoder.fc = nn.Identity()

        self.encoder = encoder
        self.backbone_family = "resnet2d"
        self.variant = variant
        self.stage_depths = _RESNET_STAGE_DEPTHS[variant]
        self.num_layers_for_decay = 5

        last_block = encoder.layer4[-1]
        in_dim = int(
            getattr(getattr(last_block, "bn3", None), "num_features", getattr(last_block.bn2, "num_features"))
        )
        self.proj = nn.Identity() if in_dim == self.feature_dim else nn.Linear(in_dim, self.feature_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(x)
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        x = self.encoder.avgpool(x)
        return torch.flatten(x, 1)

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

    def get_layer_id(self, param_name: str) -> int:
        name = param_name.replace("_fsdp_wrapped_module.", "")
        if name.startswith("backbone."):
            name = name[len("backbone.") :]

        if name.startswith("encoder.conv1") or name.startswith("encoder.bn1"):
            return 0
        if name.startswith("encoder.layer1"):
            return 1
        if name.startswith("encoder.layer2"):
            return 2
        if name.startswith("encoder.layer3"):
            return 3
        if name.startswith("encoder.layer4"):
            return 4
        return self.num_layers_for_decay
