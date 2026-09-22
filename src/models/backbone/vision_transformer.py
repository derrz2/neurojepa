"""Vision Transformer backbone for channel-time (C, T) inputs."""

import math
from functools import partial
from typing import Callable, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from timm.layers import LayerType, Mlp, get_act_layer, get_norm_layer, trunc_normal_
from timm.models.vision_transformer import get_init_weights_vit, named_apply

from .base import cfg_get
from ..layer.ffn import build_ffn_layer
from ..layer.norms import build_norm_layer
from ..layer.vit_components import Block, PatchEmbed
from ..layer.rope_position_embedding import RopePositionEmbedding
from ...utils.optim import fuse_params_groups, get_params_groups_with_decay
from ...utils.utils import _to_2tuple


class VisionTransformer(nn.Module):
    """ViT backbone for channel-time inputs."""

    dynamic_img_size: Final[bool]

    def __init__(
        self,
        args=None,
        img_size: Union[int, Tuple[int, int]] = (400, 40),
        patch_size: Union[int, Tuple[int, int]] = (20, 4),
        num_classes: int = 1000,
        global_pool: Literal['', 'avg', 'avgmax', 'max', 'token', 'map'] = 'token',
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        proj_bias: bool = True,
        init_values: Optional[float] = None,
        class_token: bool = True,
        pos_embed: str = 'learn',
        no_embed_class: bool = False,
        reg_tokens: int = 0,
        pre_norm: bool = False,
        final_norm: bool = True,
        fc_norm: Optional[bool] = None,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.3,
        weight_init: Literal['skip', 'jax', 'jax_nlhb', 'moco', ''] = '',
        fix_init: bool = False,
        embed_norm_layer: Optional[LayerType] = None,
        norm_layer: Optional[LayerType] = nn.LayerNorm,
        act_layer: Optional[LayerType] = nn.GELU,
        norm_cfg: Optional[dict] = None,
        block_fn: Type[nn.Module] = Block,
        mlp_layer: Type[nn.Module] = Mlp,
        ffn_layer: Optional[str] = None,
        downstream: Optional[bool] = True,
        fusion_mode: str = 'none',
        gate_attention: str = 'none',
        freeze_backbone: bool = False,
        rope_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()

        assert global_pool in ('', 'avg', 'avgmax', 'max', 'token', 'map')
        assert class_token or global_pool != 'token'
        assert pos_embed in ('', 'none', 'learn')

        self.img_size = _to_2tuple(img_size)
        self.patch_size = _to_2tuple(patch_size)
        self.cfg = args
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.embed_dim = embed_dim
        self.feature_dim = embed_dim
        self.downstream = bool(downstream)
        self.backbone_family = "vit"
        training_cfg = cfg_get(args, "training", None)
        activation_checkpoint_cfg = cfg_get(training_cfg, "activation_checkpointing", False)

        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.no_embed_class = no_embed_class

        rope_enable = bool(self._get_cfg(rope_cfg, "enable", False))
        rope_layer_start = int(self._get_cfg(rope_cfg, "layer_start", 0))
        rope_layer_end = int(self._get_cfg(rope_cfg, "layer_end", depth))
        rope_layer_start = max(0, min(depth, rope_layer_start))
        rope_layer_end = max(rope_layer_start, min(depth, rope_layer_end))
        self.rope_enable = rope_enable
        self.rope_layer_start = rope_layer_start
        self.rope_layer_end = rope_layer_end
        self.rope = None
        if self.rope_enable:
            self.rope = RopePositionEmbedding(
                embed_dim=embed_dim,
                num_heads=num_heads,
                rotary_dim_ratio=float(self._get_cfg(rope_cfg, "rotary_dim_ratio", 0.5)),
                base=self._get_cfg(rope_cfg, "base", 100.0),
                min_period=self._get_cfg(rope_cfg, "min_period", None),
                max_period=self._get_cfg(rope_cfg, "max_period", None),
                normalize_coords=self._get_cfg(rope_cfg, "normalize_coords", "separate"),
                shift_coords=self._get_cfg(rope_cfg, "shift_coords", None),
                jitter_coords=self._get_cfg(rope_cfg, "jitter_coords", None),
                rescale_coords=self._get_cfg(rope_cfg, "rescale_coords", None),
            )

        use_fc_norm = global_pool in ('avg', 'avgmax', 'max') if fc_norm is None else fc_norm
        if norm_cfg is not None:
            norm_eps = float(self._get_cfg(norm_cfg, "eps", 1e-6))
            norm_layer = build_norm_layer(
                norm_type=self._get_cfg(norm_cfg, "type", "layernorm"),
                eps=norm_eps,
            )
            qk_norm_layer = self._get_cfg(norm_cfg, "qk_type", "layernorm")
            qk_norm_layer = (
                norm_layer
                if str(qk_norm_layer).lower() == "inherit"
                else build_norm_layer(norm_type=qk_norm_layer, eps=norm_eps)
            )
        else:
            norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
            qk_norm_layer = norm_layer
        embed_norm_layer = get_norm_layer(embed_norm_layer)
        act_layer = get_act_layer(act_layer) or nn.GELU
        if ffn_layer is not None:
            mlp_layer = build_ffn_layer(ffn_layer)

        embed_args = {}
        if embed_norm_layer is not None:
            embed_args['norm_layer'] = embed_norm_layer

        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            embed_dim=embed_dim,
            bias=not pre_norm,
            strict_img_size=False,
            dynamic_img_pad=True,
            **embed_args,
        )
        self.base_grid_size = self.patch_embed.grid_size
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None

        embed_len = self.num_patches if no_embed_class else self.num_patches + self.num_prefix_tokens
        if not pos_embed or pos_embed == 'none':
            self.pos_embed = None
        else:
            self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)

        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        self.depth = depth
        self.activation_checkpointing, self.activation_checkpoint_blocks = self._resolve_activation_checkpointing(
            activation_checkpoint_cfg,
            depth,
        )
        self.stage_depths = [1 for _ in range(depth)]
        self.num_layers_for_decay = depth + 1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    proj_bias=proj_bias,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    qk_norm_layer=qk_norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                    gate_attention=gate_attention,
                    use_rope=(
                        self.rope_enable and (self.rope_layer_start <= i < self.rope_layer_end)
                    ),
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim) if final_norm and not use_fc_norm else nn.Identity()

        if self.downstream:
            self.fusion_mode = fusion_mode
            if self.fusion_mode == '1':
                self.layer_weights = nn.Parameter(torch.ones(8))
                self.fusion_dim = embed_dim * 5
            elif fusion_mode == '2':
                self.fusion_dim = embed_dim * 4
            elif fusion_mode == '3':
                self.layer_weights = nn.Parameter(torch.ones(4))
                self.fusion_dim = embed_dim
            else:
                self.fusion_dim = self.embed_dim

            if self.fusion_mode != 'none' and self.fusion_dim != self.embed_dim:
                self.bottennet = nn.Sequential(
                    nn.Linear(self.fusion_dim, self.embed_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                )

            self.fc_norm = norm_layer(self.embed_dim) if final_norm and use_fc_norm else nn.Identity()
            self.head_drop = nn.Dropout(drop_rate)
            if not freeze_backbone:
                self.head = nn.Sequential(
                    nn.Linear(self.embed_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, num_classes),
                )
            else:
                self.head = nn.Linear(self.embed_dim, num_classes)

        if weight_init != 'skip':
            self.init_weights(weight_init)
        if fix_init:
            self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def init_weights(self, mode: str = '') -> None:
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.0
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        if self.reg_token is not None:
            nn.init.normal_(self.reg_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    @staticmethod
    def _get_cfg(cfg_obj, key, default):
        if cfg_obj is None:
            return default
        if isinstance(cfg_obj, dict):
            return cfg_obj.get(key, default)
        if hasattr(cfg_obj, "get"):
            return cfg_obj.get(key, default)
        return getattr(cfg_obj, key, default)

    def _add_prefix_tokens(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        prefix = []
        if self.cls_token is not None:
            prefix.append(self.cls_token.expand(b, -1, -1))
        if self.reg_token is not None:
            prefix.append(self.reg_token.expand(b, -1, -1))
        if not prefix:
            return x
        prefix_tokens = torch.cat(prefix, dim=1)
        return torch.cat([prefix_tokens, x], dim=1)

    def _resize_pos_embed(self, grid_size: Tuple[int, int]) -> Optional[torch.Tensor]:
        if self.pos_embed is None:
            return None

        if tuple(grid_size) == tuple(self.base_grid_size):
            return self.pos_embed

        prefix_n = 0 if self.no_embed_class else self.num_prefix_tokens
        if prefix_n > 0:
            prefix_pos = self.pos_embed[:, :prefix_n]
            patch_pos = self.pos_embed[:, prefix_n:]
        else:
            prefix_pos = None
            patch_pos = self.pos_embed

        patch_pos = patch_pos.reshape(1, self.base_grid_size[0], self.base_grid_size[1], self.embed_dim)
        patch_pos = patch_pos.permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos,
            size=grid_size,
            mode="bicubic",
            align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], self.embed_dim)

        if prefix_pos is None:
            return patch_pos
        return torch.cat([prefix_pos, patch_pos], dim=1)

    def _split_pos_embed(self, grid_size: Tuple[int, int]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        pos_embed = self._resize_pos_embed(grid_size)
        if pos_embed is None:
            return None, None

        prefix_n = 0 if self.no_embed_class else self.num_prefix_tokens
        if prefix_n > 0:
            return pos_embed[:, :prefix_n], pos_embed[:, prefix_n:]
        return None, pos_embed

    def _gather_patch_tokens(
        self,
        x: torch.Tensor,
        keep_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if keep_mask is None:
            return x, None

        if keep_mask.ndim == 1:
            keep_mask = keep_mask.unsqueeze(0).expand(x.shape[0], -1)
        elif keep_mask.ndim != 2:
            raise ValueError(f"Expected keep_mask to have shape (N,) or (B, N), got {tuple(keep_mask.shape)}")

        keep_mask = keep_mask.to(device=x.device, dtype=torch.bool)
        keep_counts = keep_mask.sum(dim=1)
        if int(keep_counts.min().item()) != int(keep_counts.max().item()):
            raise ValueError("All samples must keep the same number of patches for token selection.")

        num_keep = int(keep_counts[0].item())
        if num_keep <= 0:
            raise ValueError("keep_mask must keep at least one patch token.")

        keep_indices = keep_mask.nonzero(as_tuple=False)[:, 1].view(x.shape[0], num_keep)
        gather_index = keep_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        return torch.gather(x, dim=1, index=gather_index), keep_indices

    def _apply_pos_embed(self, x: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
        pos_embed = self._resize_pos_embed(grid_size)
        if self.no_embed_class:
            if pos_embed is not None:
                x = x + pos_embed
            x = self._add_prefix_tokens(x)
        else:
            x = self._add_prefix_tokens(x)
            if pos_embed is not None:
                x = x + pos_embed
        return self.pos_drop(x)

    def forward_tokens(
        self,
        x: torch.Tensor,
        keep_mask: Optional[torch.Tensor] = None,
        return_grid: bool = False,
        return_keep_indices: bool = False,
    ):
        target_dtype = self.pos_embed.dtype if self.pos_embed is not None else self.patch_embed.proj.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(dtype=target_dtype)

        x, grid_size = self.patch_embed(x)
        prefix_pos, patch_pos = self._split_pos_embed(grid_size)
        if patch_pos is not None:
            x = x + patch_pos
        x, keep_indices = self._gather_patch_tokens(x, keep_mask)

        prefix_tokens = []
        if self.cls_token is not None:
            prefix_tokens.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            prefix_tokens.append(self.reg_token.expand(x.shape[0], -1, -1))
        if prefix_tokens:
            prefix_tokens = torch.cat(prefix_tokens, dim=1)
            if prefix_pos is not None:
                prefix_tokens = prefix_tokens + prefix_pos
            x = torch.cat([prefix_tokens, x], dim=1)

        x = self.pos_drop(x)
        all_layer_feats = self.forward_features(x, grid_size)
        tokens = all_layer_feats[-1]

        outputs = [tokens[:, self.num_prefix_tokens:] if self.num_prefix_tokens > 0 else tokens]
        if return_grid:
            outputs.append(grid_size)
        if return_keep_indices:
            outputs.append(keep_indices)
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def _get_rope_inputs(
        self, x: torch.Tensor, grid_size: Tuple[int, int]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.rope_enable or self.rope is None:
            return None, None, None

        b, n, _ = x.shape
        grid_c, grid_t = grid_size
        patch_n = grid_c * grid_t
        prefix_n = self.num_prefix_tokens
        if n != patch_n + prefix_n:
            return None, None, None

        c_coord = torch.arange(grid_c, device=x.device, dtype=x.dtype)
        t_coord = torch.arange(grid_t, device=x.device, dtype=x.dtype)
        cc, tt = torch.meshgrid(c_coord, t_coord, indexing='ij')
        patch_coords = torch.stack([cc.reshape(-1), tt.reshape(-1)], dim=-1)
        patch_coords = patch_coords.unsqueeze(0).expand(b, -1, -1)

        if prefix_n > 0:
            prefix_coords = torch.zeros(b, prefix_n, 2, device=x.device, dtype=x.dtype)
            coords = torch.cat([prefix_coords, patch_coords], dim=1)
            rope_mask = torch.cat(
                [
                    torch.zeros(b, prefix_n, device=x.device, dtype=x.dtype),
                    torch.ones(b, patch_n, device=x.device, dtype=x.dtype),
                ],
                dim=1,
            )
        else:
            coords = patch_coords
            rope_mask = torch.ones(b, patch_n, device=x.device, dtype=x.dtype)

        rope_sin, rope_cos = self.rope(coords, spatial_shape=grid_size)
        return rope_sin, rope_cos, rope_mask

    def _expand_key_padding_mask(
        self,
        key_padding_mask: Optional[torch.Tensor],
        num_patch_tokens: int,
    ) -> Optional[torch.Tensor]:
        if key_padding_mask is None:
            return None
        if key_padding_mask.ndim != 2:
            raise ValueError(
                f"Expected key_padding_mask with shape (B, N) or (B, N_patch), got {tuple(key_padding_mask.shape)}."
            )

        key_padding_mask = key_padding_mask.to(dtype=torch.bool)
        if key_padding_mask.shape[1] == num_patch_tokens + self.num_prefix_tokens:
            return key_padding_mask
        if key_padding_mask.shape[1] != num_patch_tokens:
            raise ValueError(
                f"Expected key_padding_mask width {num_patch_tokens} or {num_patch_tokens + self.num_prefix_tokens}, "
                f"got {key_padding_mask.shape[1]}."
            )
        if self.num_prefix_tokens <= 0:
            return key_padding_mask

        prefix_mask = torch.zeros(
            key_padding_mask.shape[0],
            self.num_prefix_tokens,
            device=key_padding_mask.device,
            dtype=torch.bool,
        )
        return torch.cat([prefix_mask, key_padding_mask], dim=1)

    def forward_features(
        self,
        x: torch.Tensor,
        grid_size: Tuple[int, int],
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        x = self.norm_pre(x)
        rope_sin, rope_cos, rope_mask = self._get_rope_inputs(x, grid_size)
        all_layer_outputs = []
        for i, block in enumerate(self.blocks):
            if self._should_checkpoint_block(i) and self.training and x.requires_grad:
                x = self._checkpoint_block(
                    x,
                    block,
                    key_padding_mask=key_padding_mask,
                    rope_sin=rope_sin,
                    rope_cos=rope_cos,
                    rope_mask=rope_mask,
                )
            else:
                x = block(
                    x,
                    key_padding_mask=key_padding_mask,
                    rope_sin=rope_sin,
                    rope_cos=rope_cos,
                    rope_mask=rope_mask,
                )
            curr_out = self.norm(x) if i == len(self.blocks) - 1 else x
            all_layer_outputs.append(curr_out)
        return all_layer_outputs

    def _checkpoint_block(
        self,
        x: torch.Tensor,
        block: nn.Module,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        def _forward(hidden_states: torch.Tensor) -> torch.Tensor:
            return block(
                hidden_states,
                key_padding_mask=key_padding_mask,
                rope_sin=rope_sin,
                rope_cos=rope_cos,
                rope_mask=rope_mask,
            )

        try:
            return torch.utils.checkpoint.checkpoint(_forward, x, use_reentrant=False)
        except TypeError:
            return torch.utils.checkpoint.checkpoint(_forward, x)

    @staticmethod
    def _normalize_checkpoint_block_idx(idx: int, depth: int) -> Optional[int]:
        if idx < 0:
            idx += depth
        if 0 <= idx < depth:
            return idx
        return None

    def _resolve_activation_checkpointing(
        self,
        raw_cfg,
        depth: int,
    ) -> Tuple[bool, Optional[set[int]]]:
        if isinstance(raw_cfg, bool):
            return raw_cfg, None

        if isinstance(raw_cfg, int):
            if raw_cfg <= 0:
                return False, None
            block_count = min(depth, raw_cfg)
            return True, set(range(depth - block_count, depth))

        if isinstance(raw_cfg, (list, tuple, set)):
            blocks = {
                normalized_idx
                for normalized_idx in (
                    self._normalize_checkpoint_block_idx(int(block_idx), depth)
                    for block_idx in raw_cfg
                )
                if normalized_idx is not None
            }
            return bool(blocks), blocks or None

        return bool(raw_cfg), None

    def _should_checkpoint_block(self, block_idx: int) -> bool:
        if not self.activation_checkpointing:
            return False
        if self.activation_checkpoint_blocks is None:
            return True
        return block_idx in self.activation_checkpoint_blocks

    def _extract_pooled_features(self, x_out_list: List[torch.Tensor]) -> List[torch.Tensor]:
        pooled_layers = []
        for x_out in x_out_list:
            if self.global_pool == 'token':
                if self.num_prefix_tokens < 1:
                    pooled = x_out.mean(dim=1)
                else:
                    pooled = x_out[:, 0]
            else:
                patch_tokens = x_out[:, self.num_prefix_tokens:] if self.num_prefix_tokens > 0 else x_out
                if patch_tokens.shape[1] == 0:
                    pooled = x_out.mean(dim=1)
                elif self.global_pool == 'avg':
                    pooled = patch_tokens.mean(dim=1)
                elif self.global_pool == 'max':
                    pooled = patch_tokens.amax(dim=1)
                elif self.global_pool == 'avgmax':
                    pooled = 0.5 * (patch_tokens.mean(dim=1) + patch_tokens.amax(dim=1))
                else:
                    pooled = patch_tokens.mean(dim=1)
            pooled_layers.append(pooled)
        return pooled_layers

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        fusion_mode = getattr(self, 'fusion_mode', 'none')
        if fusion_mode != 'none' and fusion_mode != 'pool_mean' and self.fusion_dim != self.embed_dim:
            x = self.bottennet(x)
        fc_norm = getattr(self, 'fc_norm', None)
        if fc_norm is not None:
            x = fc_norm(x)
        return x

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_embedding(x)
        x = self.head_drop(x)
        return self.head(x)

    def extract_probe_features(self, x: torch.Tensor) -> torch.Tensor:
        target_dtype = self.pos_embed.dtype if self.pos_embed is not None else self.patch_embed.proj.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(dtype=target_dtype)

        x, grid_size = self.patch_embed(x)
        x = self._apply_pos_embed(x, grid_size)

        all_layer_feats = self.forward_features(x, grid_size)
        pooled_feats = self._extract_pooled_features(all_layer_feats)

        final_feat = pooled_feats[-1]
        fusion_mode = getattr(self, 'fusion_mode', 'none')
        if fusion_mode == '1':
            norm_weights = F.softmax(self.layer_weights, dim=0)
            low_level_feat = 0
            for i in range(8):
                low_level_feat = low_level_feat + norm_weights[i] * pooled_feats[i]
            high_level_feat = torch.cat(pooled_feats[8:], dim=-1)
            final_feat = torch.cat([low_level_feat, high_level_feat], dim=-1)
        elif fusion_mode == '2':
            final_feat = torch.cat(pooled_feats[8:], dim=-1)
        elif fusion_mode == '3':
            norm_weights = F.softmax(self.layer_weights, dim=0)
            last4_feats = pooled_feats[-4:]
            feat = 0
            for i in range(4):
                feat = feat + norm_weights[i] * last4_feats[i]
            final_feat = feat

        return self.forward_embedding(final_feat)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_all_layers: bool = False,
        return_tokens: bool = False,
        return_probe_features: bool = False,
    ) -> torch.Tensor:
        target_dtype = self.pos_embed.dtype if self.pos_embed is not None else self.patch_embed.proj.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(dtype=target_dtype)

        x, grid_size = self.patch_embed(x)
        key_padding_mask = self._expand_key_padding_mask(key_padding_mask, x.shape[1])
        x = self._apply_pos_embed(x, grid_size)

        all_layer_feats = self.forward_features(x, grid_size, key_padding_mask=key_padding_mask)
        if return_tokens:
            tokens = all_layer_feats[-1]
            return tokens[:, self.num_prefix_tokens:] if self.num_prefix_tokens > 0 else tokens
        pooled_feats = self._extract_pooled_features(all_layer_feats)

        if return_all_layers:
            return pooled_feats
        if not self.downstream and not return_probe_features:
            return pooled_feats[-1]

        final_feat = pooled_feats[-1]
        fusion_mode = getattr(self, 'fusion_mode', 'none')
        if fusion_mode == '1':
            norm_weights = F.softmax(self.layer_weights, dim=0)
            low_level_feat = 0
            for i in range(8):
                low_level_feat = low_level_feat + norm_weights[i] * pooled_feats[i]
            high_level_feat = torch.cat(pooled_feats[8:], dim=-1)
            final_feat = torch.cat([low_level_feat, high_level_feat], dim=-1)
        elif fusion_mode == '2':
            final_feat = torch.cat(pooled_feats[8:], dim=-1)
        elif fusion_mode == '3':
            norm_weights = F.softmax(self.layer_weights, dim=0)
            last4_feats = pooled_feats[-4:]
            feat = 0
            for i in range(4):
                feat = feat + norm_weights[i] * last4_feats[i]
            final_feat = feat

        embedding = self.forward_embedding(final_feat)
        if return_probe_features:
            return embedding
        logits = self.head_drop(embedding)
        logits = self.head(logits)
        return logits

    def get_params_groups(self):
        params_groups = get_params_groups_with_decay(
            model=self,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)

        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups

    def get_layer_id(self, param_name: str) -> int:
        name = param_name.replace("_fsdp_wrapped_module.", "")
        if name.startswith("backbone."):
            name = name[len("backbone.") :]

        if (
            "pos_embed" in name
            or "patch_embed" in name
            or "mask_token" in name
            or "cls_token" in name
            or "register_tokens" in name
            or "mixed_patch" in name
        ):
            return 0

        if ".blocks." in name and ".residual." not in name:
            return int(name[name.find(".blocks.") :].split(".")[2]) + 1
        if name.startswith("blocks.") and ".residual." not in name:
            return int(name.split(".")[1]) + 1

        return self.num_layers_for_decay
