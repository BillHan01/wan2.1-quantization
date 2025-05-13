from typing import Any, Dict, Optional
import torch.cuda.amp as amp
import torch
import torch.nn.functional as F
from torch import nn

from timm.models.vision_transformer import Mlp
import xformers.ops
from viditq_extension.nn.base import QuantParams
from viditq_extension.nn.qlinear import W8A8OF16LinearDynamicInputScale
from viditq_extension.nn.layernorm import LayerNormGeneral
import viditq_extension.fused as fused_kernels

import logging
logger = logging.getLogger(__name__)

from .modules.model import WanLayerNorm, WanRMSNorm
from .modules.attention import flash_attention

import time

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
    print(f"  [int_weight] shape: {int_weight.shape}, dtype: {int_weight.dtype}")
    print(f"  [int_weight] max: {int_weight.max().item()}, min: {int_weight.min().item()}")
    # print(f"  [DONE] {full_name} quantized and saved.\n")

from .modules.model import (
    WanSelfAttention,
    WanT2VCrossAttention,
    # Attention,
    # CaptionEmbedder,
    # MultiHeadCrossAttention,
    # PatchEmbed3D,
    # PositionEmbedding2D,
    # SeqParallelAttention,
    # SeqParallelMultiHeadCrossAttention,
    # SizeEmbedder,
    # T2IFinalLayer,
    # TimestepEmbedder,
    # approx_gelu,
    # get_layernorm,
    # t2i_modulate,
    # LlamaRMSNorm,
)

from timm.models.layers import DropPath 
from einops import rearrange


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanAttentionBlockWithCudaKernel(nn.Module):
    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 quant_params=None,
            ):
        super().__init__()

        self.quant_params = quant_params

        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # use_kernel: idx 0 - SelfAttn; 1 - CrossAttn; 2 - MLP
        self.use_kernel = [True, False, False]

        # layers
        # layer 0: selfAttention

        if self.use_kernel[0]:
            self.norm1 = LayerNormGeneral(dim, act_sum=True, eps=eps)
            self.self_attn = WanSelfAttentionWithCudaKernel(dim, num_heads, window_size, qk_norm, eps, quant_params=quant_params)
        else:
            self.norm1 = WanLayerNorm(dim, eps)
            self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)

        # layer 1: crossAttention
        if self.use_kernel[1]:
            self.norm3 = LayerNormGeneral(dim, act_sum=True, eps=eps)
            self.cross_attn = WANT2VCrossAttentionWithCudaKernel(dim, num_heads, (-1, -1), qk_norm, eps, quant_params=quant_params)
        else:
            self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
            self.cross_attn = WanT2VCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)

        # layer 2: MLP
        if self.use_kernel[2]:
            self.norm2 = LayerNormGeneral(dim, act_sum=True, eps=eps)
            # self.ffn = FFNWithCudaKernel(dim=dim, ffn_dim=ffn_dim, quant_params=quant_params)
        else:
            self.norm2 = WanLayerNorm(dim, eps)
            self.ffn = nn.Sequential(
                nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
                nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)


    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        torch.cuda.synchronize()
        t0 = time.time()

        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
            # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = e 
        assert e[0].dtype == torch.float32
        # logger.info(f"[Block] input mean: {x.mean().item()}, std: {x.std().item()}, dtype: {x.dtype}")  
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
            t.to(torch.float16) for t in e
        ]

        torch.cuda.synchronize()
        print(f"calculate e took {time.time() - t0:.4f} seconds")
        torch.cuda.synchronize()
        t0 = time.time()

        # self-attention
        if self.use_kernel[0]:
            # print("quant self-attention")
            x = x.to(dtype=torch.float16).contiguous()       # kernel needs fp16 and contiguous input

            torch.cuda.synchronize()
            t1 = time.time()

            sa_input = self.norm1(x, shift_msa, scale_msa, self.quant_params)   # fp16 -> int8
            
            torch.cuda.synchronize()
            print(f"norm1   {time.time() - t1:.4f} seconds")
            torch.cuda.synchronize()
            t1 = time.time()

            y = self.self_attn(sa_input, seq_lens, grid_sizes, freqs, e)        # int8 -> fp32
            torch.cuda.synchronize()
            print(f"sa   {time.time() - t1:.4f} seconds")

            torch.cuda.synchronize()
            t1 = time.time()

            y = y.to(dtype=torch.float16)  
            residual = x
            x = fused_kernels.gate_residual_fuse(
                y.view(-1, y.shape[-1]), 
                gate_msa.view(-1, gate_msa.shape[-1]), 
                residual.view(-1, residual.shape[-1])
            ).view(y.shape)

            torch.cuda.synchronize()
            print(f"residual   {time.time() - t1:.4f} seconds")
            # logger.info(f"[DEBUG] after gate_residual_fuse (MSA): mean={x.mean().item():.6f}, std={x.std().item():.6f}")
        else:
            sa_input = self.norm1(x).float() * (1 + e[1]) + e[0]
            y = self.self_attn(sa_input, seq_lens, grid_sizes, freqs)
            with amp.autocast(dtype=torch.float32):
                x = x + y * e[2]
        
        torch.cuda.synchronize()
        print(f"Self-Attention block took {time.time() - t0:.4f} seconds in total")
        torch.cuda.synchronize()
        t0 = time.time()
        # cross-attention
        if self.use_kernel[1]:
            # print("quant cross-attention")
            x = x.to(dtype=torch.float16).contiguous()  
            ca_x = self.norm3(x, shift_msa, scale_msa, self.quant_params)
            x = x + self.cross_attn(ca_x, context, context_lens)  # to modify...
        else:
            x = x + self.cross_attn(self.norm3(x), context, context_lens)

        # logger.info(f"[DEBUG] after cross_attn: mean={x.mean().item():.6f}, std={x.std().item():.6f}")        
        torch.cuda.synchronize()
        print(f"Cross-Attention block took {time.time() - t1:.4f} seconds in total")
        torch.cuda.synchronize()
        t0 = time.time()

        # ffn
        if self.use_kernel[2]:
            x = x.to(dtype=torch.float16).contiguous()       # kernel needs fp16 and contiguous input
            # print(f"[DEBUG] input shape: {x.shape}")
            # print(f"[DEBUG] quant_params.sum_input shape: {self.quant_params.sum_input.shape}")
            # print(f"[DEBUG] quant_params.scaling shape: {self.quant_params.scale_input.shape}")
            ffn_input = self.norm2(x, shift_msa, scale_msa, self.quant_params)
            # print(f"ffn_input dtype: {ffn_input.dtype}")
            # logger.info(f"[DEBUG] after norm2 (ffn_input): mean={ffn_input.float().mean().item():.6f}, std={ffn_input.float().std().item():.6f}")

            y = self.ffn(ffn_input)
            # logger.info(f"[DEBUG] after ffn: mean={y.mean().item():.6f}, std={y.std().item():.6f}")
            y = y.to(dtype=torch.float16).contiguous()       # kernel needs fp16 and contiguous input
            residual = x
            x = fused_kernels.gate_residual_fuse(
                y.view(-1, y.shape[-1]), 
                gate_mlp.view(-1, gate_mlp.shape[-1]), 
                residual.view(-1, residual.shape[-1])
            ).view(y.shape)
            # logger.info(f"[DEBUG] after gate_residual_fuse (FFN): mean={x.mean().item():.6f}, std={x.std().item():.6f}")            
            # with amp.autocast(dtype=torch.float32):
            #     x = x + y * e[5]
        else:
            # print("don't quant ffn")
            torch.cuda.synchronize()
            t1 = time.time()

            y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
            with amp.autocast(dtype=torch.float32):
                x = x + y * e[5]

            torch.cuda.synchronize()
            print(f"FFN took {time.time() - t0:.4f} seconds in total")
            


        # cross-attention & ffn function
        # def cross_attn_ffn(x, context, context_lens, e):
        #     x = x + self.cross_attn(self.norm3(x), context, context_lens)
        #     y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
        #     with amp.autocast(dtype=torch.float32):
        #         x = x + y * e[5]
        #     return x
        # 
        # x = cross_attn_ffn(x, context, context_lens, e)

        return x

# padding tools
def pad_to_multiple_2d(x, multiple_L=128, multiple_C=64):
    B, L, C = x.shape
    pad_L = (multiple_L - L % multiple_L) % multiple_L
    pad_C = (multiple_C - C % multiple_C) % multiple_C
    # Pad last dimension (C) first
    if pad_C > 0:
        x = F.pad(x, (0, pad_C), value=0)  # pad (dim=-1)
    if pad_L > 0:
        x = F.pad(x, (0, 0, 0, pad_L), value=0)  # pad (dim=-2)
    return x, pad_L, pad_C

def pad_quant_param(qparam: torch.Tensor, pad_len: int, value=0.):
    if pad_len == 0:
        return qparam
    pad_tensor = torch.full((pad_len,), value, dtype=qparam.dtype, device=qparam.device)
    return torch.cat([qparam, pad_tensor], dim=0)


class WanSelfAttentionWithCudaKernel(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 quant_params=None,
                 weight_sym=False,
                ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        self.quant_params = quant_params
        self.weight_sym = weight_sym    

        # layers
        # self.q = nn.Linear(dim, dim)
        self.q = W8A8OF16LinearDynamicInputScale(dim, dim, has_bias=True, weight_sym=self.weight_sym)
        self.k = W8A8OF16LinearDynamicInputScale(dim, dim, has_bias=True, weight_sym=self.weight_sym)
        self.v = W8A8OF16LinearDynamicInputScale(dim, dim, has_bias=True, weight_sym=self.weight_sym)
        self.linear_k = nn.Linear(dim, dim)
        self.linear_v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor, seq_lens, grid_sizes, freqs, e) -> torch.Tensor:
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        # b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        b, s, n, d = x.shape[0], x.shape[1], self.num_heads, self.head_dim

        # # padding
        # def pad_to_multiple_2d(x, multiple_L=128, multiple_C=64):
        #     B, L, C = x.shape
        #     pad_L = (multiple_L - L % multiple_L) % multiple_L
        #     pad_C = (multiple_C - C % multiple_C) % multiple_C
        #     # Pad last dimension (C) first
        #     if pad_C > 0:
        #         x = F.pad(x, (0, pad_C), value=0)  # pad (dim=-1)
        #     if pad_L > 0:
        #         x = F.pad(x, (0, 0, 0, pad_L), value=0)  # pad (dim=-2)
        #     return x, pad_L, pad_C

        # def pad_quant_param(qparam: torch.Tensor, pad_len: int, value=0.):
        #     if pad_len == 0:
        #         return qparam
        #     pad_tensor = torch.full((pad_len,), value, dtype=qparam.dtype, device=qparam.device)
        #     return torch.cat([qparam, pad_tensor], dim=0)

        torch.cuda.synchronize()
        t0 = time.time()
        x, pad_L, pad_C = pad_to_multiple_2d(x, 128, 64)
        quant_params = QuantParams(
            seq_len=self.quant_params.scale_input.shape[0],
            has_sum_input=self.quant_params.has_sum_input,
            device=self.quant_params.scale_input.device,
        )
        quant_params.scale_input = self.quant_params.scale_input.clone()
        quant_params.sum_input = self.quant_params.sum_input.clone()     
        quant_params.scale_input = pad_quant_param(self.quant_params.scale_input, pad_L * b, value=1.)
        quant_params.sum_input = pad_quant_param(self.quant_params.sum_input, pad_L * b, value=0.)

        torch.cuda.synchronize()    
        print(f"padding {time.time() - t0:.4f} seconds")  

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
            t.to(torch.float16) for t in e
        ]        

        # torch.cuda.synchronize()
        # t0 = time.time()

        # q = self.q(x, quant_params)
        # torch.cuda.synchronize()    
        # print(f"q {time.time() - t0:.4f} seconds")
        torch.cuda.synchronize()
        t0 = time.time()

        q = self.q(x, quant_params)

        torch.cuda.synchronize()    
        print(f"q {time.time() - t0:.4f} seconds")       
        torch.cuda.synchronize()
        t0 = time.time()

        q = self.norm_q(q)

        torch.cuda.synchronize()    
        print(f"norm q {time.time() - t0:.4f} seconds")        

        q = q[:, :s, :]
        q = q.view(b, s, n, d)

        k = self.norm_k(self.k(x, quant_params) )
        k = k[:, :s, :]
        k = k.view(b, s, n, d)

        v = self.v(x, quant_params)
        v = v[:, :s, :]        
        v = v.view(b, s, n, d)
        
   
        torch.cuda.synchronize()
        t0 = time.time()

        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)       

        torch.cuda.synchronize()    
        print(f"rope apply {time.time() - t0:.4f} seconds")    

        x = x[:, :s, :]

        torch.cuda.synchronize()
        t0 = time.time()

        x = flash_attention(
            q=q,
            k=k,
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        torch.cuda.synchronize()    
        print(f"flash attention {time.time() - t0:.4f} seconds")    
        # output
        x = x.flatten(2)
        x = self.o(x)
        # print(f">>>>>>> x = self.o(x) done!")  

        return x


class WANT2VCrossAttentionWithCudaKernel(WanSelfAttentionWithCudaKernel):
    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, s, n, d = x.size(0), x.size(1), self.num_heads, self.head_dim

        # padding
        x, pad_L_x, pad_C_x = pad_to_multiple_2d(x, 128, 64)
        # quant params of padded x
        quant_params = QuantParams(
            seq_len=self.quant_params.scale_input.shape[0],
            has_sum_input=self.quant_params.has_sum_input,
            device=self.quant_params.scale_input.device,
        )
        quant_params.scale_input = self.quant_params.scale_input.clone()
        quant_params.sum_input = self.quant_params.sum_input.clone()     
        quant_params.scale_input = pad_quant_param(self.quant_params.scale_input, pad_L_x * b, value=1.)
        quant_params.sum_input = pad_quant_param(self.quant_params.sum_input, pad_L_x * b, value=0.)

        # compute query, key, value
        f_q = self.q(x, quant_params)
        print(f"f_q shape: {f_q.shape}")
        f_q = f_q[:, :s, :]  # unpad
        print(f"unpadded f_q shape: {f_q.shape}")
        q = self.norm_q(f_q)
        q = q.view(b, -1, n, d)

        k = self.norm_k(self.linear_k(context)).view(b, -1, n, d)
        v = self.linear_v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


# ffn 
# class FFNWithCudaKernel(nn.Module):
#     def __init__(
#         self,
#         dim,
#         ffn_dim,
#         weight_sym=False,
#         quant_params=None,
#     ):
#         super().__init__()
#         self.fc1 = W8A8OF16LinearDynamicInputScale(dim, ffn_dim, has_bias=True, weight_sym=weight_sym)
#         self.fc2 = nn.Linear(ffn_dim, dim)
#         # self.fc2 = W8A8OF16LinearDynamicInputScale(ffn_dim, dim, has_bias=True, weight_sym=weight_sym)   
#         self.gelu = nn.GELU(approximate='tanh')     
#         self.quant_params = quant_params

#     def forward(self, x):
#         b, s = x.shape[0], x.shape[1]
#         print(f"[DEBUG] berfore pad input shape: {x.shape}, should be 128x")
#         x, pad_L, pad_C = pad_to_multiple_2d(x, 128, 64)
#         quant_params = QuantParams(
#             seq_len=self.quant_params.scale_input.shape[0],
#             has_sum_input=self.quant_params.has_sum_input,
#             device=self.quant_params.scale_input.device,
#         )
#         quant_params.scale_input = self.quant_params.scale_input.clone()
#         quant_params.sum_input = self.quant_params.sum_input.clone()     
#         quant_params.scale_input = pad_quant_param(self.quant_params.scale_input, pad_L * b, value=1.)
#         quant_params.sum_input = pad_quant_param(self.quant_params.sum_input, pad_L * b, value=0.)
#         print(f"[DEBUG] kernel input shape: {x.shape}, should be 128x")
#         x = self.fc1(x, quant_params)
#         print(f"[DEBUG] hidden size: {x.shape[-1]}, should be <= 8192, x mean: {x.mean().item()}, std: {x.std().item()}")
#         hidden_size = x.shape[-1]
#         if hidden_size <= 8192:
#             x = fused_kernels.gelu_quant_sum(x, quant_params.sum_input, quant_params.scale_input)
#         else:
#             # hidden_size is too large, then skip applying quant
#             print(f"[WARNING] hidden_size {hidden_size} > 8192, skipping quantized gelu, using float gelu instead.")
#             x = self.gelu(x)  # normal gelu
#         x = x[:, :s, :]
#         print(f"gelu dtype: {x.dtype}, shape: {x.shape}, mean: {x.mean().item()}, std: {x.std().item()}")
#         # x = fused_kernels.quant_sum(x, self.quant_params.sum_input, self.quant_params.scale_input)
#         x = self.fc2(x)

#         return x 