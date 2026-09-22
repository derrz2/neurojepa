"""Build the ViT architecture used by the released NeuroJEPA encoders."""
from copy import deepcopy

from .base import cfg_get
from vit_size_presets import VIT_SIZE_PRESETS


def build_backbone(args, *, downstream=None, num_classes=None):
    model_cfg = cfg_get(args, "model", args)
    backbone_name = str(cfg_get(model_cfg, "backbone_name", None) or
                        cfg_get(model_cfg, "backbone_family", None) or "vit").lower()
    if backbone_name != "vit":
        raise ValueError(f"This release supports only the ViT backbone, got: {backbone_name}")
    model_size = str(cfg_get(model_cfg, "model_size", None) or "base").lower()
    cfg_copy = args if downstream is None and num_classes is None else deepcopy(args)
    cfg_model = cfg_get(cfg_copy, "model", cfg_copy)
    if downstream is not None:
        cfg_model["downstream"] = bool(downstream)
    if num_classes is not None:
        cfg_model["num_classes"] = int(num_classes)

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
