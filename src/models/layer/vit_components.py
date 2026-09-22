from typing import Callable, List, Optional, Tuple, Type, Union
from ...utils.utils import _to_2tuple
import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.layers import Mlp, DropPath
import torch.nn.init as init
try:
    from flash_attn import flash_attn_varlen_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


class PatchEmbed(nn.Module):
    """Plain patch embedding for CxT inputs via Conv2d."""

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = (400, 40),
        patch_size: Union[int, Tuple[int, int]] = (20, 4),
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten: bool = True,
        bias: bool = True,
        strict_img_size: bool = True,
        dynamic_img_pad: bool = False,
    ) -> None:
        super().__init__()
        self.img_size = _to_2tuple(img_size)
        self.patch_size = _to_2tuple(patch_size)
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, C, T), got {tuple(x.shape)}")

        _, c, t = x.shape
        if self.strict_img_size:
            if c != self.img_size[0] or t != self.img_size[1]:
                raise ValueError(
                    f"Input (C,T)=({c},{t}) does not match configured img_size={self.img_size}."
                )

        pad_c = (-c) % self.patch_size[0]
        pad_t = (-t) % self.patch_size[1]

        x = x.unsqueeze(1)  # (B, 1, C, T)
        if pad_c > 0 or pad_t > 0:
            if not self.dynamic_img_pad:
                raise ValueError(
                    f"Input (C,T)=({c},{t}) is not divisible by patch_size={self.patch_size}."
                )
            x = F.pad(x, (0, pad_t, 0, pad_c))

        x = self.proj(x)  # (B, D, C', T')
        grid_size = (x.shape[-2], x.shape[-1])
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        x = self.norm(x)
        return x, grid_size

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    x_rot = torch.stack((-x_odd, x_even), dim=-1)
    return x_rot.flatten(-2)


def _apply_rotary_emb(
    x: torch.Tensor,
    rope_sin: Optional[torch.Tensor],
    rope_cos: Optional[torch.Tensor],
    rope_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if rope_sin is None or rope_cos is None:
        return x
    rot_dim = rope_sin.shape[-1]
    if rot_dim <= 0:
        return x

    rot_dim = min(rot_dim, x.shape[-1])
    rot_dim = (rot_dim // 2) * 2
    if rot_dim <= 0:
        return x

    x_rope = x[..., :rot_dim]
    x_tail = x[..., rot_dim:]

    sin = rope_sin[..., :rot_dim].to(dtype=x.dtype, device=x.device)
    cos = rope_cos[..., :rot_dim].to(dtype=x.dtype, device=x.device)

    if x.ndim == 3:
        # x: (Total, H, D)
        sin = sin.unsqueeze(1)
        cos = cos.unsqueeze(1)
    elif x.ndim == 4:
        # x: (B, H, N, D), rope: (B, N, D)
        sin = sin.unsqueeze(1)
        cos = cos.unsqueeze(1)
    else:
        raise ValueError(f"Unsupported tensor rank for RoPE: {x.ndim}")

    x_rot = x_rope * cos + _rotate_half(x_rope) * sin

    if rope_mask is not None:
        mask = rope_mask.to(dtype=x.dtype, device=x.device)
        if x.ndim == 3:
            mask = mask[:, None, None]
        else:
            mask = mask[:, None, :, None]
        x_rope = x_rot * mask + x_rope * (1.0 - mask)
    else:
        x_rope = x_rot

    return torch.cat([x_rope, x_tail], dim=-1)


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            qk_norm_layer: Optional[Type[nn.Module]] = None,
            gate_attention: str = 'none'
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.gate_attention = gate_attention

        self.q_aug_dim = 0
        if gate_attention == 'headwise':
            self.q_aug_dim = self.num_heads
        elif gate_attention == 'elementwise':
            self.q_aug_dim = self.dim
        total_dim = dim * 3 + self.q_aug_dim

        self.qkv = nn.Linear(dim, total_dim, bias=qkv_bias)
        qk_norm_layer = qk_norm_layer or norm_layer
        self.q_norm = qk_norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = qk_norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            x: torch.Tensor,
            cu_seqlens: Optional[torch.Tensor] = None,
            max_seqlen: Optional[int] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
            rope_sin: Optional[torch.Tensor] = None,
            rope_cos: Optional[torch.Tensor] = None,
            rope_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        if torch.isnan(x).any():
             print("!!! [Attention Input] x contains NaN!")

        is_packed = x.ndim == 2
        if is_packed:
            Total, C = x.shape
            B = 1 # Dummy
            N = Total
        else:
            B, N, C = x.shape

        qkv = self.qkv(x)

        if torch.isnan(qkv).any():
             print("!!! [QKV Output] contains NaN! (Weights might be corrupted)")
        
        # qkv shape: (B, N, Total_Dim) or (Total, Total_Dim)
        
        q_aug, k, v = torch.split(qkv, [self.dim + self.q_aug_dim, self.dim, self.dim], dim=-1)

        if self.gate_attention == 'headwise':
            q, gate = torch.split(q_aug, [self.dim, self.num_heads], dim=-1)
            # Gate: (..., num_heads) -> (..., num_heads, 1)
            gate = gate.unsqueeze(-1) 
        elif self.gate_attention == 'elementwise':
            q, gate = torch.split(q_aug, [self.dim, self.dim], dim=-1)
            # Gate: (..., dim) -> (..., num_heads, head_dim)
            gate = gate.reshape(*gate.shape[:-1], self.num_heads, self.head_dim)
        else:
            q = q_aug
            gate = None

        # --- Flash Attention Path (Varlen) ---
        if HAS_FLASH_ATTN and cu_seqlens is not None and max_seqlen is not None:

            cu_seqlens = cu_seqlens.to(dtype=torch.int32).contiguous()
            if torch.isnan(q).any() or torch.isinf(q).any():
                print("!!! Q contains NaN/Inf BEFORE FlashAttn")

            # Reshape Q, K, V to (Total, Num_Heads, Head_Dim)
            q = q.reshape(-1, self.num_heads, self.head_dim)
            k = k.reshape(-1, self.num_heads, self.head_dim)
            v = v.reshape(-1, self.num_heads, self.head_dim)
            if self.gate_attention == 'headwise':
                q = q.contiguous()
                k = k.contiguous()
                v = v.contiguous()

            if torch.isnan(q).any(): print("!!! Q is NaN BEFORE Norm")

            if gate is not None:
                if self.gate_attention == 'headwise':
                    gate = gate.reshape(-1, self.num_heads, 1)
                else:
                    gate = gate.reshape(-1, self.num_heads, self.head_dim)

            q, k = self.q_norm(q), self.k_norm(k)
            q = _apply_rotary_emb(q, rope_sin=rope_sin, rope_cos=rope_cos, rope_mask=rope_mask)
            k = _apply_rotary_emb(k, rope_sin=rope_sin, rope_cos=rope_cos, rope_mask=rope_mask)

            if torch.isnan(q).any(): 
                print(f"!!! Q contains NaN AFTER Norm. Q_Norm Weight: {self.q_norm.weight.mean()}")

            # Cast to bf16/fp16 for flash attention
            target_dtype = q.dtype
            if q.dtype == torch.float32:
                target_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
                q = q.to(target_dtype)
                k = k.to(target_dtype)
                v = v.to(target_dtype)

            cu_seqlens = cu_seqlens.to(torch.int32)
            
            # Flash Attention Call
            out = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=False
            )
            
            if out.dtype != x.dtype:
                out = out.to(x.dtype)

            if gate is not None:
                out = out * F.sigmoid(gate)
            
            # Reshape back to input shape
            if is_packed:
                out = out.reshape(Total, C)
            else:
                out = out.reshape(B, N, C)

        # --- Standard Attention Path ---
        else:
            if is_packed:
                raise ValueError("Standard Attention does not support packed sequences directly. Use (B, N, C) input or enable Flash Attention.")

            attn_mask = None
            if key_padding_mask is not None:
                attn_mask = torch.zeros_like(key_padding_mask, dtype=q.dtype)
                attn_mask.masked_fill_(key_padding_mask, float("-inf"))
                # Reshape for broadcasting: (B, Num_Heads, Q_len, K_len)
                attn_mask = attn_mask.view(B, 1, 1, N)

            q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            if self.gate_attention == 'headwise':
                q = q.contiguous()
                k = k.contiguous()
                v = v.contiguous()

            q, k = self.q_norm(q), self.k_norm(k)
            q = _apply_rotary_emb(q, rope_sin=rope_sin, rope_cos=rope_cos, rope_mask=rope_mask)
            k = _apply_rotary_emb(k, rope_sin=rope_sin, rope_cos=rope_cos, rope_mask=rope_mask)
            
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )

            out = out.transpose(1, 2) # (B, N, H, D)
            
            if gate is not None:
                # Gate: (B, N, H, 1) or (B, N, H, D) -> Permute -> (B, H, N, D/1)
                # gate = gate.permute(0, 2, 1, 3)
                out = out * F.sigmoid(gate)
                
            out = out.reshape(B, N, C)

        x = self.proj(out)
        x = self.proj_drop(x)
        return x

class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            qk_norm_layer: Optional[Type[nn.Module]] = None,
            mlp_layer: Type[nn.Module] = Mlp,
            gate_attention: str = 'none',
            use_rope: bool = False
    ) -> None:
        super().__init__()
        self.use_rope = use_rope
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            qk_norm_layer=qk_norm_layer,
            gate_attention=gate_attention
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            bias=proj_bias,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
            self,
            x: torch.Tensor,
            cu_seqlens: Optional[torch.Tensor] = None,
            max_seqlen: Optional[int] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
            rope_sin: Optional[torch.Tensor] = None,
            rope_cos: Optional[torch.Tensor] = None,
            rope_mask: Optional[torch.Tensor] = None,
            **kwargs
    ) -> torch.Tensor:
        if not self.use_rope:
            rope_sin, rope_cos, rope_mask = None, None, None
        x = x + self.drop_path1(self.ls1(self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            key_padding_mask=key_padding_mask,
            rope_sin=rope_sin,
            rope_cos=rope_cos,
            rope_mask=rope_mask
        )))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, norm_layer=nn.BatchNorm1d, act_layer=nn.ReLU):
        super().__init__()
        layers = []
        dims = [in_features] + hidden_features
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            if i < len(dims) - 2:
                if norm_layer is not None:
                    layers.append(norm_layer(dims[i + 1]))
                layers.append(act_layer())
        
        self.model = nn.Sequential(*layers)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            init.ones_(m.weight)
            init.zeros_(m.bias)

    def forward(self, x):
        for layer in self.model:
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                input_dtype = x.dtype
                x = layer(x.float())
                if input_dtype != torch.float32:
                    x = x.to(input_dtype)
            else:
                x = layer(x)
        return x

