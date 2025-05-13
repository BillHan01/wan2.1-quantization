import xformers.ops
from einops import rearrange

import torch
import sys
import os
import argparse
import numpy as np
from omegaconf import OmegaConf

import torch.nn as nn
import torch.nn.functional as F
from qdiff.base.base_quantizer import StaticQuantizer, DynamicQuantizer, BaseQuantizer
from qdiff.base.quant_layer import QuantizedLinear
from qdiff.utils import apply_func_to_submodules
from qdiff.base.quant_model import quant_layer_refactor_, bitwidth_refactor_, load_quant_param_dict_, save_quant_param_dict_, set_init_done_
from qdiff.base.quant_attn import QuantizedAttentionMapOpenSORA

# from models.quant_opensora_cuda import quantize_and_save_weight_, STDiT3BlockWithCudaKernel
import logging
logger = logging.getLogger(__name__)

from .modules.model import WanModel
from diffusers.configuration_utils import register_to_config
from .quant_wanx_cuda import quantize_and_save_weight_, WanAttentionBlockWithCudaKernel


class QuantWanModel(WanModel):

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=1536,
                 ffn_dim=8960,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=12,
                 num_layers=30,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 quant_config=None
            ):

        super().__init__(
            model_type=model_type,
            patch_size=patch_size,
            text_len=text_len,
            in_dim=in_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            window_size=window_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
        )

        # load_checkpoint(self, from_pretrained)
        self.quant_config=quant_config
        self.quant_param_dict = {}

        # print(">>>>>>> self.quant_layer_refactor()")
        # self.quant_layer_refactor()


    def convert_quant(self, quant_config):
        self.quant_config = quant_config
        self.quant_param_dict = {}
        self.quant_layer_refactor()

    def quant_layer_refactor(self):
        '''
        INFO: always replace the Attn & CrossAttn,
        due to sometimes we need to FP infer the Quantized module for apply_hooks
        '''
        # replace the linear layers
        apply_func_to_submodules(self,
                class_type=nn.Linear,
                function=quant_layer_refactor_,
                name=None,
                parent_module=None,
                quant_config=self.quant_config,
                full_name=None,
                remain_fp_regex=self.quant_config.remain_fp_regex,
                )
    
    def save_quant_param_dict(self):
        apply_func_to_submodules(self,
                class_type=BaseQuantizer,
                function=save_quant_param_dict_,
                full_name=None,
                parent_module=None,
                model=self
                )

    def load_quant_param_dict(self, quant_param_dict):
        apply_func_to_submodules(self,
                class_type=BaseQuantizer,
                function=load_quant_param_dict_,
                full_name=None,
                parent_module=None,
                quant_param_dict=quant_param_dict,
                model=self,
                )

    def set_init_done(self):
        apply_func_to_submodules(self,
                class_type=BaseQuantizer,
                function=set_init_done_,)
        
    def bitwidth_refactor(self):
        apply_func_to_submodules(self,
                class_type=QuantizedLinear,
                function=bitwidth_refactor_,
                name=None,
                parent_module=None,
                quant_config=self.quant_config,
                full_name=None
                )


    # # ------ used for infer with CUDA kernel ------- 
    def quantize_and_save_weight(self, save_path):
        # set require_grad=False, since torch force the variable to be FP or complex (we assign them as torch.int8)
        for param in self.parameters():
            param.requires_grad_(False)

        # iter through all QuantLayers and get quantized INT, fill into the state_dict
        apply_func_to_submodules(self,
                class_type=QuantizedLinear,
                function=quantize_and_save_weight_, # =====> to modify...
                full_name=None,
                )

        # delete the quant_params and fp_weights in the state_dict
        sd = self.state_dict()
        keys_to_delete = ['fp_weight','fp_module']  # INFO: the ptq process, when `sym=True` the weight quant has 0. zero_point, so delete them.
        keys_to_rename = {
                'w_quantizer.delta': 'scale_weight',
                'w_quantizer.zero_point': 'zp_weight',
                }
        for k in list(sd.keys()):
            if any(substring in k for substring in keys_to_delete):
                # print('delete key: {}'.format(k))
                del sd[k]
            if any(substring in k for substring in keys_to_rename):
                original_k = k
                for substring in keys_to_rename.keys():
                    if substring in k:
                        # print('rename key: {} to {}'.format(k, keys_to_rename[substring]))
                        k = k.replace(substring, keys_to_rename[substring])
                sd[k] = sd.pop(original_k)       
                
        # torch.cuda.empty_cache() 

        # INFO: we implement the "general" version of layernorm
        # with the affine transform (wx+b) fused after the layernorm operation
        # so the vanilla layernorm has w=1 and b=0   
        n_block = len(self.blocks)
        for i_block in range(n_block):
            hidden_size = self.blocks[0].norm1.normalized_shape[0]
            sd['blocks.{}.norm1.weight'.format(i_block)] = torch.ones((hidden_size,), dtype=torch.float16)
            sd['blocks.{}.norm2.weight'.format(i_block)] = torch.ones((hidden_size,), dtype=torch.float16) 
        
        # logger.info('Remaining Keys')
        # for k in list(sd.keys()):
        #     logger.info('key: {},  {}'.format(k, sd[k].dtype))
        print(">>>>>>>> save key")
        torch.save(sd, save_path)

        logger.info("\nFinished Saving the Quantized Checkpoint...\n")


    def hardware_forward_refactor(self, load_path, seq_len):
        from viditq_extension.nn.base import QuantParams

        # (1) Set the seq_len to init the QuantParams 
        # ----- Wanx Transformer H x W -------
        # target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
        #                 size[1] // self.vae_stride[1],
        #                 size[0] // self.vae_stride[2])
        # seq_len = math.ceil((target_shape[2] * target_shape[3]) /
        #                     (self.patch_size[1] * self.patch_size[2]) *
        #                     target_shape[1] / self.sp_size) * self.sp_size
        print("seq_len: ", seq_len)
        self.quant_params = QuantParams(seq_len, has_sum_input=True, device=torch.device("cuda"))

        # (2) replace the blocks, with cuda kernel version    
        print("replace the blocks with cuda kernel version")    
        n_block = len(self.blocks)
        for i_block in range(n_block):     
            old_block = self.blocks[i_block]

            self.blocks[i_block] = WanAttentionBlockWithCudaKernel(
                dim=old_block.dim,
                ffn_dim=old_block.ffn_dim,
                num_heads=old_block.num_heads,
                window_size=old_block.window_size,
                qk_norm=old_block.qk_norm,
                cross_attn_norm=old_block.cross_attn_norm,
                eps=old_block.eps,
                quant_params=self.quant_params,
            ).half().to('cuda')
            setattr(self.blocks[i_block],'block_id',i_block)
            

        # (3) load the integer weights
        print("load the quantized weights")
        quant_sd = torch.load(load_path, weights_only=True, map_location='cuda')
        self.load_state_dict(quant_sd, strict=False)
        print("load the quantized weights done")
        for k, v in quant_sd.items():
            if "weight" in k or "scale_weight" in k:
                logger.info(f"[Load Check] {k}: min={v.min().item()}, max={v.max().item()}, mean={v.float().mean().item()}, dtype={v.dtype}")
