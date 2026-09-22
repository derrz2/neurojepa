from __future__ import annotations

from copy import deepcopy

from .base import cfg_get
from vit_size_presets import VIT_SIZE_PRESETS


_RESNET_SIZE_TO_VARIANT = {
    "tiny": "resnet2d_18",
    "small": "resnet2d_34",
    "base": "resnet2d_50",
    "large": "resnet2d_101",
    "huge": "resnet2d_152",
}


_CONVNEXT_SIZE_TO_VARIANT = {
    "tiny": "convnext2d_tiny",
    "small": "convnext2d_small",
    "base": "convnext2d_base",
    "large": "convnext2d_large",
    "huge": "convnext2d_xlarge",
}


_INCEPTION2D_SIZE_PRESETS = {
    "tiny": {"dims": [64, 128, 256], "stage_depths": [1, 2, 1]},
    "small": {"dims": [96, 192, 384], "stage_depths": [2, 2, 2]},
    "base": {"dims": [128, 256, 512], "stage_depths": [2, 3, 2]},
    "large": {"dims": [160, 320, 640], "stage_depths": [3, 4, 3]},
    "huge": {"dims": [192, 384, 768], "stage_depths": [3, 5, 4]},
}


def _resolve_backbone_name(args) -> str:
    model_cfg = cfg_get(args, "model", args)
    name = cfg_get(model_cfg, "backbone_name", None)
    family = cfg_get(model_cfg, "backbone_family", None)
    if name:
        return str(name)
    if family:
        return str(family)
    return "vit"


def _resolve_model_size(args) -> str:
    model_cfg = cfg_get(args, "model", args)
    size = cfg_get(model_cfg, "model_size", None)
    if size is not None:
        return str(size).lower()

    backbone_name = _resolve_backbone_name(args).lower()
    if backbone_name == "vit":
        return "base"
    if backbone_name in {"resnet2d_18", "resnet18"}:
        return "tiny"
    if backbone_name in {"resnet2d_34", "resnet34"}:
        return "small"
    if backbone_name in {"resnet2d_50", "resnet50"}:
        return "base"
    if backbone_name in {"resnet2d_101", "resnet101"}:
        return "large"
    if backbone_name in {"resnet2d_152", "resnet152"}:
        return "huge"
    if backbone_name in {"convnext2d_tiny", "convnext_tiny"}:
        return "tiny"
    if backbone_name in {"convnext2d_small", "convnext_small"}:
        return "small"
    if backbone_name in {"convnext2d_base", "convnext_base"}:
        return "base"
    if backbone_name in {"convnext2d_large", "convnext_large"}:
        return "large"
    if backbone_name in {"convnext2d_xlarge", "convnext_xlarge"}:
        return "huge"
    if backbone_name in {"inception2d", "inception2d_small"}:
        return "small"
    return "base"


def build_backbone(args, *, downstream=None, num_classes=None):
    model_cfg = cfg_get(args, "model", args)
    backbone_name = _resolve_backbone_name(args).lower()
    model_size = _resolve_model_size(args)
    cfg_copy = args if downstream is None and num_classes is None else deepcopy(args)
    cfg_model = cfg_get(cfg_copy, "model", cfg_copy)

    if downstream is not None:
        cfg_model["downstream"] = bool(downstream)
    if num_classes is not None:
        cfg_model["num_classes"] = int(num_classes)

    if backbone_name == "vit":
        try:
            from .vision_transformer import VisionTransformer
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "ViT backbone requires the optional dependency `timm` to be installed."
            ) from exc
        preset = VIT_SIZE_PRESETS.get(model_size, VIT_SIZE_PRESETS["base"])
        if cfg_get(cfg_model, "embed_dim", None) is None:
            cfg_model["embed_dim"] = preset["embed_dim"]
        if cfg_get(cfg_model, "depth", None) is None:
            cfg_model["depth"] = preset["depth"]
        if cfg_get(cfg_model, "num_heads", None) is None:
            cfg_model["num_heads"] = preset["num_heads"]
        return VisionTransformer(
            img_size=cfg_get(model_cfg, "img_size", (100, 200)),
            patch_size=cfg_get(model_cfg, "patch_size", (10, 10)),
            global_pool=cfg_get(model_cfg, "global_pool", "token"),
            embed_dim=cfg_get(cfg_model, "embed_dim", 768),
            depth=cfg_get(cfg_model, "depth", 12),
            num_heads=cfg_get(cfg_model, "num_heads", 12),
            mlp_ratio=cfg_get(model_cfg, "mlp_ratio", 4.0),
            qkv_bias=cfg_get(model_cfg, "qkv_bias", True),
            qk_norm=cfg_get(model_cfg, "qk_norm", True),
            drop_path_rate=cfg_get(model_cfg, "drop_path_rate", 0.0),
            init_values=cfg_get(model_cfg, "init_values", None),
            downstream=cfg_get(cfg_model, "downstream", False),
            num_classes=cfg_get(cfg_model, "num_classes", 1000),
            reg_tokens=int(cfg_get(model_cfg, "reg_tokens", 0)),
            gate_attention=cfg_get(model_cfg, "gate_attention", "none"),
            pos_embed=cfg_get(model_cfg, "pos_embed", "learn"),
            rope_cfg=cfg_get(model_cfg, "rope", None),
            norm_cfg=cfg_get(model_cfg, "norm", None),
            ffn_layer=cfg_get(model_cfg, "ffn_layer", None),
            args=cfg_copy,
            drop_rate=cfg_get(model_cfg, "drop_rate", 0.0),
            attn_drop_rate=cfg_get(model_cfg, "attn_drop_rate", 0.0),
        )

    if backbone_name.startswith("resnet2d") or backbone_name.startswith("resnet"):
        from .resnet2d import ResNet2DBackbone
        return ResNet2DBackbone(
            cfg_copy,
            variant=_RESNET_SIZE_TO_VARIANT.get(model_size, "resnet2d_50"),
            downstream=downstream,
            num_classes=num_classes,
        )

    if backbone_name.startswith("convnext2d") or backbone_name.startswith("convnext"):
        from .convnext2d import ConvNeXt2DBackbone
        return ConvNeXt2DBackbone(
            cfg_copy,
            variant=_CONVNEXT_SIZE_TO_VARIANT.get(model_size, "convnext2d_base"),
            downstream=downstream,
            num_classes=num_classes,
        )

    if backbone_name.startswith("inception2d"):
        from .inception2d import Inception2DBackbone
        preset = _INCEPTION2D_SIZE_PRESETS.get(model_size, _INCEPTION2D_SIZE_PRESETS["base"])
        cfg_model["dims"] = list(preset["dims"])
        cfg_model["stage_depths"] = list(preset["stage_depths"])
        return Inception2DBackbone(cfg_copy, variant=f"inception2d_{model_size}", downstream=downstream, num_classes=num_classes)

    raise ValueError(f"Unsupported backbone_name: {backbone_name}")
