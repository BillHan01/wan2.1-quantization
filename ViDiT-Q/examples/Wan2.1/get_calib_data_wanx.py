# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
from datetime import datetime
import logging
import os
import sys
import warnings

warnings.filterwarnings('ignore')

import torch, random
import torch.nn as nn
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video, cache_image, str2bool
import time
import requests
from PIL import Image
from io import BytesIO


from qdiff.utils import apply_func_to_submodules, seed_everything
from qdiff.base.quant_layer import QuantizedLinear


EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2v-14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2i-14B": {
        "prompt": "一个朴素端庄的美人",
    },
    "i2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
}



def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 40 if "i2v" in args.task else 50

    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832"]:
            args.sample_shift = 3.0

    # The default number of frames are 1 for text-to-image tasks and 81 for other tasks.
    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81

    # T2I frame_num check
    if "t2i" in args.task:
        assert args.frame_num == 1, f"Unsupport frame_num {args.frame_num} for task {args.task}"

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames to sample from a image or video. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the image or video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="ch",
        choices=["ch", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.")

    args = parser.parse_args()

    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


class SaveActivationHook:
    def __init__(self, type=None, original_shape=None):
        self.hook_handle = None
        self.type = type
        self.original_shape = original_shape
        self.outputs = []
        self.attn_ds_rate = None
        
    def attn_map_downsample(self, data):
        '''
        down_sample in the N_token dimension, handle the indivisible situation. 
        '''
        assert self.type == 'attn'
        BS, head_per_split_num, N_token, N_token = self.original_shape
        N_remainder = N_token % self.attn_ds_rate
        data = data[:,:,:-N_remainder,:-N_remainder]
        data = data.reshape([
            BS,head_per_split_num,N_token//self.attn_ds_rate,self.attn_ds_rate,N_token//self.attn_ds_rate,self.attn_ds_rate
            ])
        return data.max(dim=3)[0].max(dim=4)[0]
        
    def __call__(self, module, module_in, module_out):
        '''
        the input shape could be [BS, N_group];
        reduce along the head dimension. 
        '''
        if self.type == 'qk':
            BS, head_per_split_num, N_token, N_dim = self.original_shape
            data = module_in[0].reshape(self.original_shape).to('cpu')
            # data = module_in[0].reshape(self.original_shape).abs().max(dim=-1)[0].to('cpu') # avoid taking up too much GPU memory
        elif self.type == 'v':
            BS, head_per_split_num, N_dim, N_token = self.original_shape
            data = module_in[0].reshape(self.original_shape).to('cpu')
        elif self.type == 'attn':
            BS, head_per_split_num, N_token, N_token = self.original_shape
            data = module_in[0].reshape(self.original_shape).to('cpu')
            if self.attn_ds_rate is not None:
                data = self.attn_map_downsample(data)
        elif self.type == 'cross_attn':
            BS, head_per_split_num, N_image_token, N_text_token = self.original_shape
            data = module_in[0].reshape(self.original_shape).to('cpu')
            # no attn_downsample for cross_attn, it wont be that big. 
        else:
            C = module_in[0].shape[-1]
            data = module_in[0].reshape([-1,C]).abs().max(dim=0)[0]  # for smooth quant
            #raise NotImplementedError
        
        # TODO: maybe post processing. 
        self.outputs.append(data)

    def clear(self):
        self.outputs = []

def add_hook_to_module_(module, hook_cls, **kwargs):
    hook = hook_cls(**kwargs)
    hook.hook_handle = module.register_forward_hook(hook)
    return hook


def generate(args):

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (initialize_model_parallel,
                                             init_distributed_environment)
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )
    
    cfg = WAN_CONFIGS[args.task]
    # all_prompts = [args.prompt]
    if "t2v" in args.task or "t2i" in args.task:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        # logging.info(f"Input prompt: {args.prompt}")
        if args.use_prompt_extend:
            print("use_prompt_extend")

        logging.info("Creating WanT2V pipeline.")
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )
    else:
        logging.info("Creating WanI2V pipeline.")
        wan_i2v = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )
    
    print(">>>>>>>>>>>> add the hook")
    # ------------ Add the Hooks for Calib Data --------------
    kwargs = {
        'hook_cls': SaveActivationHook,
    }
    hook_d = apply_func_to_submodules(wan_t2v.model,
                            class_type=nn.Linear,  # add hook to all objects of this cls
                            function=add_hook_to_module_,
                            return_d={},
                            **kwargs
                            )
    print("hook added to wan!")

    with open("/cv/hanjiarui/code/ViDiT-Q/examples/Wan2.1/test_prompts.txt", "r") as f:
        all_prompts = f.readlines()

    for prompt_idx, prompt_tmp in enumerate(all_prompts):
            
        args.prompt = prompt_tmp

        if args.use_prompt_extend:
            if args.prompt_extend_method == "dashscope":
                prompt_expander = DashScopePromptExpander(
                    model_name=args.prompt_extend_model, is_vl="i2v" in args.task)
            elif args.prompt_extend_method == "local_qwen":
                prompt_expander = QwenPromptExpander(
                    model_name=args.prompt_extend_model,
                    is_vl="i2v" in args.task,
                    device=rank)
            else:
                raise NotImplementedError(
                    f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

        cfg = WAN_CONFIGS[args.task]
        if args.ulysses_size > 1:
            assert cfg.num_heads % args.ulysses_size == 0, f"`num_heads` must be divisible by `ulysses_size`."

        logging.info(f"Generation job args: {args}")
        logging.info(f"Generation model config: {cfg}")

        if dist.is_initialized():
            base_seed = [args.base_seed] if rank == 0 else [None]
            dist.broadcast_object_list(base_seed, src=0)
            args.base_seed = base_seed[0]

        if "t2v" in args.task or "t2i" in args.task:
            if args.prompt is None:
                args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
            logging.info(f"Input prompt: {args.prompt}")

            logging.info(
                f"Generating {'image' if 't2i' in args.task else 'video'} ...")
            video = wan_t2v.generate(
                args.prompt,
                size=SIZE_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed,
                offload_model=args.offload_model)
            
            torch.cuda.empty_cache()
        else:
            return

        # time.sleep(2)
        torch.cuda.empty_cache()

        logging.info(f"rank  is {rank}")
    logging.info("Finished.")

    print(">>>>>>>>>>>> Unpack the hooked results")
    # --------- Unpack the hooked results and save  -------------
    # save_d = {}
    # for k,v in hook_d.items():
    #     # when using fsdp for multi GPU，firstly delete '_fsdp_wrapped_module.' prefix
    #     print(f"original key name: {k}")
    #     clean_key = k.replace("_fsdp_wrapped_module.", "")
    #     # print(f"    =====> {k}")
    #     save_d[clean_key] = torch.stack(v.outputs, dim=0)  # [N_timestep, C]
    #     logging.info(f'layer_name: {clean_key}, hook_input_shape: {v.outputs[0].shape}')
    #     v.hook_handle.remove()

    print(">>>>>>>>>>>> to save calib data")
    gather_and_save_activation(hook_d, "/cv/hanjiarui/code/ViDiT-Q/examples/Wan2.1/quant_data/calib_data_wanx1.pth")
    # torch.save(save_d, quant_config.calib_data.save_path)
    # torch.save(save_d, "/cv/hanjiarui/code/ViDiT-Q/examples/Wan2.1/quant_data/calib_data_wanx1.pth")
    print(">>>>>>>>>>>> calib data saved")


def gather_and_save_activation(hook_d, save_path):
    # 把 hook_d 变成可序列化的 dict，清理掉不可保存的字段
    save_d = {}
    for k, v in hook_d.items():
        clean_key = k.replace("_fsdp_wrapped_module.", "")
        outputs = torch.stack(v.outputs, dim=0).cpu()
        save_d[clean_key] = outputs  # shape: [N_timestep, C]

        if hasattr(v, 'hook_handle'):
            v.hook_handle.remove()  # 移除 hook，释放资源

    # 获取 rank/world_size
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # 分布式收集所有 rank 的 save_d
    all_d_list = [None for _ in range(world_size)]
    dist.all_gather_object(all_d_list, save_d)

    if rank == 0:
        # 合并所有 rank 的激活（避免重复 key 的话，这里直接更新；可加去重逻辑）
        merged = {}
        for d in all_d_list:
            for k, v in d.items():
                if k in merged:
                    merged[k] = torch.cat([merged[k], v], dim=0)  # 合并时间步
                else:
                    merged[k] = v

        print(f"[RANK 0] Saving merged calib data with {len(merged)} keys to {save_path}")
        torch.save(merged, save_path)

if __name__ == "__main__":
    args = _parse_args()
    generate(args)
