from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
import logging

from timm.models.vision_transformer import Mlp
import xformers.ops
from viditq_extension.nn.base import QuantParams
from viditq_extension.nn.qlinear import W8A8OF16LinearDynamicInputScale
from viditq_extension.nn.layernorm import LayerNormGeneral
import viditq_extension.fused as fused_kernels

import logging
logger = logging.getLogger(__name__)

# From PyTorch internals
from functools import partial
from itertools import repeat
import collections.abc
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
    
def quantize_and_save_weight_(submodule, full_name):
    fp_weight = submodule.fp_module.weight.to(torch.float16)
    # the viditq_extension.nn.qlinear use [C] as the scale shape, but the qdiff simulation code use [C, 1]

    submodule.w_quantizer.delta = submodule.w_quantizer.delta.view(-1).to(torch.float16)
    submodule.w_quantizer.zero_point = submodule.w_quantizer.zero_point.view(-1).to(torch.float16)
    scale = submodule.w_quantizer.delta
    zero_point = submodule.w_quantizer.zero_point  # the cuda kernel code uses 128+zero_point

    # INFO: the orginal module weight is the FP16 quantized dequant weight, 
    # replace with INT weight, should update the state_dict
    int_weight = torch.clamp(
            torch.round(fp_weight / scale.view(-1,1)) - zero_point.view(-1,1),
            -128, 127).to(torch.int8)  # kernel supports W8A8 only for now
    submodule.weight.data = int_weight
    
    
from opensora.models.layers.blocks import (
    Attention,
    CaptionEmbedder,
    MultiHeadCrossAttention,
    PatchEmbed3D,
    PositionEmbedding2D,
    SeqParallelAttention,
    SeqParallelMultiHeadCrossAttention,
    SizeEmbedder,
    T2IFinalLayer,
    TimestepEmbedder,
    approx_gelu,
    get_layernorm,
    t2i_modulate,
    LlamaRMSNorm,
)
from timm.models.layers import DropPath 
from einops import rearrange

class STDiT3BlockWithCudaKernel(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        rope=None,
        qk_norm=False,
        temporal=False,
        enable_flash_attn=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
        quant_params=None,
    ):
        super().__init__()
        
        self.quant_params = quant_params
        
        self.temporal = temporal
        self.hidden_size = hidden_size
        self.enable_flash_attn = enable_flash_attn
        self.enable_sequence_parallelism = enable_sequence_parallelism

        # use_kernel: idx 0 - Attention; 1 - CrossAttn; 2 - MLP
        self.use_kernel = [False, True, True] 

        if self.enable_sequence_parallelism and not temporal:
            raise AssertionError
            attn_cls = SeqParallelAttention
            mha_cls = SeqParallelMultiHeadCrossAttention
        else:
            attn_cls = AttentionWithCudaKernel if self.use_kernel[0] else Attention
            mha_cls = MultiHeadCrossAttentionWithCudaKernel if self.use_kernel[1] else MultiHeadCrossAttention
            
        if self.use_kernel[0]:
            # self.norm1 = LayerNormGeneral(hidden_size, act_sum=True, eps=1e-6)
            self.norm1 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
            self.attn = attn_cls(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                qk_norm=qk_norm,
                rope=rope,
                enable_flash_attn=enable_flash_attn,
                quant_params=self.quant_params,
            )
        else:
            self.norm1 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
            self.attn = attn_cls(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                qk_norm=qk_norm,
                rope=rope,
                enable_flash_attn=enable_flash_attn,
            )
        
        if self.use_kernel[1]:
            self.cross_attn = mha_cls(hidden_size, num_heads, quant_params=self.quant_params)
        else:
            self.cross_attn = mha_cls(hidden_size, num_heads)
            
        if self.use_kernel[2]:
            self.norm2 = LayerNormGeneral(hidden_size, act_sum=True, eps=1e-6)
            self.mlp = MlpWithCudaKernel(
                in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0,
                quant_params=self.quant_params,
            )
        else:
            self.norm2 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
            self.mlp = Mlp(
                in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0,
            )
            
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

    def t_mask_select(self, x_mask, x, masked_x, T, S):
        # x: [B, (T, S), C]
        # mased_x: [B, (T, S), C]
        # x_mask: [B, T]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        masked_x = rearrange(masked_x, "B (T S) C -> B T S C", T=T, S=S)
        x = torch.where(x_mask[:, :, None, None], x, masked_x)
        x = rearrange(x, "B T S C -> B (T S) C")
        return x

    def forward(
        self,
        x,
        y,
        t,
        mask=None,  # text mask
        x_mask=None,  # temporal mask
        t0=None,  # t with timestamp=0
        T=None,  # number of frames
        S=None,  # number of pixel patches
    ):
        # prepare modulate parameters
        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        if x_mask is not None:
            shift_msa_zero, scale_msa_zero, gate_msa_zero, shift_mlp_zero, scale_mlp_zero, gate_mlp_zero = (
                self.scale_shift_table[None] + t0.reshape(B, 6, -1)
            ).chunk(6, dim=1)
        x = x.contiguous()
            
        # attention
        if self.use_kernel[0]:
            residual = x
            x_m = self.norm1(x, shift_msa, scale_msa, self.quant_params)  
            if self.temporal:
                x_m = rearrange(x_m, "B (T S) C -> (B S) T C", T=T, S=S)
                x_m = self.attn(x_m)
                x_m = rearrange(x_m, "(B S) T C -> B (T S) C", T=T, S=S)
            else:
                x_m = rearrange(x_m, "B (T S) C -> (B T) S C", T=T, S=S)
                x_m = self.attn(x_m)
                x_m = rearrange(x_m, "(B T) S C -> B (T S) C", T=T, S=S)
            # modulate (attention)
            x = fused_kernels.gate_residual_fuse(x_m.view(-1, C), gate_msa.view(-1, C), residual.contiguous().view(-1, C)).reshape([B, N, C])
        else:
            x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
            if self.temporal:
                x_m = rearrange(x_m, "B (T S) C -> (B S) T C", T=T, S=S)
                x_m = self.attn(x_m)
                x_m = rearrange(x_m, "(B S) T C -> B (T S) C", T=T, S=S)
            else:
                x_m = rearrange(x_m, "B (T S) C -> (B T) S C", T=T, S=S)
                x_m = self.attn(x_m)
                x_m = rearrange(x_m, "(B T) S C -> B (T S) C", T=T, S=S)
            x_m_s = gate_msa * x_m
            x = x + self.drop_path(x_m_s)

        # cross attention
        if self.use_kernel[1]:    
            residual = x
            x = fused_kernels.quant_sum(x, self.quant_params.sum_input, self.quant_params.scale_input)
            x = self.cross_attn(x, y, mask)
            x = residual + x
        else:
            x = x + self.cross_attn(x, y, mask)
            
        # MLP
        if self.use_kernel[2]:
            # modulate (MLP)
            residual = x
            x = self.norm2(x, shift_mlp, scale_mlp, self.quant_params)
            x = self.mlp(x)
            x = fused_kernels.gate_residual_fuse(x.contiguous().view(-1, C), gate_mlp.view(-1, C), residual.contiguous().view(-1, C)).reshape([B, N, C])
        else:
            # modulate (MLP)
            x_m = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
            # MLP
            x_m = self.mlp(x_m)
            # modulate (MLP)
            x_m_s = gate_mlp * x_m
            # residual
            x = x + self.drop_path(x_m_s)

        return x

class AttentionWithCudaKernel(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = LlamaRMSNorm,
        enable_flash_attn: bool = False,
        rope=None,
        qk_norm_legacy: bool = False,
        quant_params=None,
        has_bias=True,
        weight_sym=False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flash_attn = enable_flash_attn

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv = W8A8OF16LinearDynamicInputScale(dim, dim * 3, has_bias=has_bias, weight_sym=weight_sym)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.qk_norm_legacy = qk_norm_legacy
        self.attn_drop = nn.Dropout(attn_drop)
        
        self.proj = W8A8OF16LinearDynamicInputScale(dim, dim, has_bias=has_bias, weight_sym=weight_sym)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = False
        if rope is not None:
            self.rope = True
            self.rotary_emb = rope
        
        self.is_causal = False
        self.quant_params = quant_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # flash attn is not memory efficient for small sequences, this is empirical
        enable_flash_attn = self.enable_flash_attn and (N > B)
        qkv = self.qkv(x, self.quant_params)
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)

        qkv = qkv.view(qkv_shape).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.qk_norm_legacy:
            # WARNING: this may be a bug
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)
            q, k = self.q_norm(q), self.k_norm(k)
        else:
            q, k = self.q_norm(q), self.k_norm(k)
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)

        if enable_flash_attn:
            from flash_attn import flash_attn_func

            # (B, #heads, N, #dim) -> (B, N, #heads, #dim)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            x = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=self.is_causal,
            )
        else:
            dtype = q.dtype
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # translate attn to float32
            attn = attn.to(torch.float32)
            if self.is_causal:
                causal_mask = torch.tril(torch.ones_like(attn), diagonal=0)
                causal_mask = torch.where(causal_mask.bool(), 0, float('-inf'))
                attn += causal_mask
            attn = attn.softmax(dim=-1)
            attn = attn.to(dtype)  # cast back attn to original dtype
            attn = self.attn_drop(attn)
            x = attn @ v

        x_output_shape = (B, N, C)
        if not enable_flash_attn:
            x = x.transpose(1, 2)
        x = x.reshape(x_output_shape)
        x = fused_kernels.quant_sum(x, self.quant_params.sum_input, self.quant_params.scale_input)
        x = self.proj(x, self.quant_params)
        x = self.proj_drop(x)
        return x
    
class MultiHeadCrossAttentionWithCudaKernel(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0, \
            quant_params=None, has_bias=True, weight_sym=False):
        super(MultiHeadCrossAttentionWithCudaKernel, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = W8A8OF16LinearDynamicInputScale(d_model, d_model, has_bias=has_bias, weight_sym=weight_sym)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.proj = W8A8OF16LinearDynamicInputScale(d_model, d_model, has_bias=has_bias, weight_sym=weight_sym)

        self.quant_params = quant_params

    def forward(self, x, cond, mask=None):
        # query/value: img tokens; key: condition; mask: if padding tokens
        B, N, C = x.shape

        q = self.q_linear(x, self.quant_params).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)

        attn_bias = None
        if mask is not None:
            attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
        x = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=attn_bias).view(B, N, C)

        x = fused_kernels.quant_sum(x, self.quant_params.sum_input, self.quant_params.scale_input)
        x = self.proj(x, self.quant_params)

        return x

# located in timm.layers.Mlp
class MlpWithCudaKernel(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
            # quant related attributes.
            weight_sym=False,
            quant_params=None,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        
        self.fc1 = W8A8OF16LinearDynamicInputScale(in_features, hidden_features, has_bias=bias[0], weight_sym=weight_sym)
        self.fc2 = W8A8OF16LinearDynamicInputScale(hidden_features, in_features, has_bias=bias[1], weight_sym=weight_sym)
        self.quant_params = quant_params

    def forward(self, x):
        x = self.fc1(x, self.quant_params)
        x = fused_kernels.gelu_quant_sum(x, self.quant_params.sum_input, self.quant_params.scale_input)
        x = self.fc2(x, self.quant_params)
        return x