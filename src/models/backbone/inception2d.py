from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn

from .base import GridBackboneBase, cfg_get


def _pair(kernel) -> Tuple[int, int]:
    if isinstance(kernel, (list, tuple)):
        return int(kernel[0]), int(kernel[1])
    k = int(kernel)
    return k, k


class ConvNormAct2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size=3, stride=1) -> None:
        super().__init__()
        kh, kw = _pair(kernel_size)
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=(kh, kw),
                stride=stride,
                padding=(kh // 2, kw // 2),
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class InceptionModule2D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        branch_ch: int,
        bottleneck_ch: int,
        kernels: Iterable[Tuple[int, int]],
        pool_proj_ch: int,
    ) -> None:
        super().__init__()
        self.bottleneck = ConvNormAct2d(in_ch, bottleneck_ch, kernel_size=1) if bottleneck_ch > 0 else nn.Identity()
        mid_ch = bottleneck_ch if bottleneck_ch > 0 else in_ch

        self.branches = nn.ModuleList(
            [ConvNormAct2d(mid_ch, branch_ch, kernel_size=kernel) for kernel in kernels]
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvNormAct2d(in_ch, pool_proj_ch, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.bottleneck(x)
        outs = [branch(base) for branch in self.branches]
        outs.append(self.pool_branch(x))
        return torch.cat(outs, dim=1)


class InceptionResidualBlock2D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        branch_kernels,
        bottleneck_ch: int,
        pool_proj_ch: int,
    ) -> None:
        super().__init__()
        n_conv_branches = len(branch_kernels)
        branch_ch = max(1, (out_ch - pool_proj_ch) // max(n_conv_branches, 1))
        fused_out_ch = branch_ch * n_conv_branches + pool_proj_ch
        self.body = InceptionModule2D(
            in_ch,
            branch_ch=branch_ch,
            bottleneck_ch=bottleneck_ch,
            kernels=branch_kernels,
            pool_proj_ch=pool_proj_ch,
        )
        self.proj = nn.Identity()
        if in_ch != fused_out_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, fused_out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(fused_out_ch),
            )
        self.act = nn.ReLU(inplace=True)
        self.out_channels = fused_out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.body(x) + self.proj(x))


class Inception2DBackbone(GridBackboneBase):
    def __init__(self, args=None, *, variant: str = "inception2d_base", downstream=None, num_classes=None) -> None:
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

        dims = list(cfg_get(model_cfg, "dims", [96, 192, 384]))
        stage_depths = list(cfg_get(model_cfg, "stage_depths", [2, 2, 2]))
        branch_kernels = [
            _pair(kernel)
            for kernel in cfg_get(model_cfg, "branch_kernels", [[3, 3], [5, 5], [7, 7]])
        ]
        bottleneck_ratio = float(cfg_get(model_cfg, "bottleneck_ratio", 0.5))
        pool_proj_ratio = float(cfg_get(model_cfg, "pool_proj_ratio", 0.25))
        stem_stride = int(cfg_get(model_cfg, "stem_stride", 2))

        self.backbone_family = "inception2d"
        self.variant = variant
        self.stage_depths = stage_depths
        self.num_layers_for_decay = len(stage_depths) + 1

        self.stem = nn.Sequential(
            ConvNormAct2d(1, dims[0], kernel_size=7, stride=stem_stride),
            ConvNormAct2d(dims[0], dims[0], kernel_size=3, stride=1),
        )

        stages = []
        downsamples = []
        curr_ch = dims[0]
        for stage_idx, (out_ch, depth) in enumerate(zip(dims, stage_depths)):
            blocks = []
            for _ in range(depth):
                bottleneck_ch = max(1, int(curr_ch * bottleneck_ratio))
                pool_proj_ch = max(1, int(out_ch * pool_proj_ratio))
                block = InceptionResidualBlock2D(
                    curr_ch,
                    out_ch,
                    branch_kernels=branch_kernels,
                    bottleneck_ch=bottleneck_ch,
                    pool_proj_ch=pool_proj_ch,
                )
                curr_ch = block.out_channels
                blocks.append(block)
            stages.append(nn.Sequential(*blocks))
            if stage_idx < len(dims) - 1:
                downsamples.append(
                    ConvNormAct2d(curr_ch, dims[stage_idx + 1], kernel_size=3, stride=2)
                )
                curr_ch = dims[stage_idx + 1]
        self.stages = nn.ModuleList(stages)
        self.downsamples = nn.ModuleList(downsamples)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Identity() if curr_ch == self.feature_dim else nn.Linear(curr_ch, self.feature_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(x)
        x = self.stem(x)
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx < len(self.downsamples):
                x = self.downsamples[idx](x)
        x = self.pool(x)
        return torch.flatten(x, 1)

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

    def get_layer_id(self, param_name: str) -> int:
        name = param_name.replace("_fsdp_wrapped_module.", "")
        if name.startswith("backbone."):
            name = name[len("backbone.") :]
        if name.startswith("stem"):
            return 0
        if name.startswith("stages."):
            return int(name.split(".")[1]) + 1
        if name.startswith("downsamples."):
            return int(name.split(".")[1]) + 1
        return self.num_layers_for_decay
