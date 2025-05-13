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

from wan.quant_wanx import QuantWanModel
from omegaconf import OmegaConf
from qdiff.utils import apply_func_to_submodules, seed_everything

from functools import partial
from wan.distributed.fsdp import shard_model
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp import FullStateDictConfig
from safetensors.torch import load_file, save_file
import json

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

    parser.add_argument(
        "--quant_config",
        type=str,
        default=None,
        help="Quant config file name.")
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


def save_checkpoint(transformer, rank, output_dir):
    print(f"--> saving checkpoint xxx")
    with FSDP.state_dict_type(
            transformer,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        cpu_state = transformer.state_dict()
    # todo move to get_state_dict
    if rank <= 0:
        save_dir = os.path.join(output_dir, f"checkpoint")
        os.makedirs(save_dir, exist_ok=True)
        # save using safetensors
        weight_path = os.path.join(save_dir, "quant_model.safetensors")
        save_file(cpu_state, weight_path)
        print(f"--> save all quant model params as safetensors")
        # config_dict = dict(transformer.config)
        # if "dtype" in config_dict:
        #     del config_dict["dtype"]  # TODO
        # config_path = os.path.join(save_dir, "config.json")
        # # save dict as json
        # with open(config_path, "w") as f:
        #     json.dump(config_dict, f, indent=4)
        
        # Save quant_param_dict separately as .pth
        quant_dict = getattr(transformer, "module", transformer).quant_param_dict
        torch.save(quant_dict, os.path.join(save_dir, "quant_params.pth"))
        print(f"--> save quant params as pth")
    print(f"--> checkpoint saved xxxx")


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
        logging.info(f"Input prompt: {args.prompt}")
        if args.use_prompt_extend:
            print("use_prompt_extend")

    print("Loading quant_config from:", args.quant_config)
    quant_config = OmegaConf.load(args.quant_config)
    print(">>>>>>>>> create model")
    # model = WanModel.from_pretrained(checkpoint_dir)
    print(args.ckpt_dir)
    model = QuantWanModel.from_pretrained(
        args.ckpt_dir,
        quant_config=quant_config)
    model.to(device)
    print(">>>>>>> self.quant_layer_refactor()")
    model.quant_layer_refactor()
    model.eval()
    # self.model.eval().requires_grad_(False)
    # print(device)

    '''
    INFO: The PTQ process:
    for simple PTQ with dynamic act quant: 
    the weight are quantized with quant_model initialization.
    the act quant params are calculated online. 
    '''
    # utilize vidit-q
    # ========> modify to multi GPU...

    def init_rotation_and_channel_mask_(module, full_name, calib_data):
        assert isinstance(module, ViDiTQuantizedLinear)
        act_mask = calib_data[full_name].max(dim=0)[0]  # [T, C], averaged over all timesteps
        print(f"act_mask.to =====> {module.fp_module.weight.device}")
        act_mask = act_mask.to(module.fp_module.weight.device)  # 移到同一设备上

        zero_mask = act_mask < 1e-3
        act_mask = torch.where(zero_mask, torch.tensor(1e-3), act_mask)
        module.get_channel_mask(act_mask)  # set self.channel_mask
        module.get_rotation_matrix()
        module.update_quantized_weight_rotated_and_scaled()

    '''
    INFO: combining both smooth_quant and quarot (vidit-q)
    '''
    if quant_config.get("viditq",None) is not None:
        from qdiff.viditq.viditq_quant_layer import ViDiTQuantizedLinear
        
        assert quant_config.calib_data.save_path is not None
        calib_data = torch.load(quant_config.calib_data.save_path, weights_only=True)  # default wtih 
        print("Collected keys:", list(calib_data.keys()))
        kwargs = {}
        apply_func_to_submodules(model,
                            class_type=ViDiTQuantizedLinear,  # add hook to all objects of this cls
                            function=init_rotation_and_channel_mask_,
                            full_name='',
                            calib_data = calib_data,
                            **kwargs
                            )        

    shard_fn = partial(shard_model, device_id=device)
    model = shard_fn(model)

    model.set_init_done()
    model.save_quant_param_dict()

    # fsdp:
     
    # model.to(device)
    print(model)
    print(f"[Device {device}] Applying FSDP to model...")
    # torch.save(model.quant_param_dict, os.path.join(cfg.save_dir, 'quant_params.pth'))
    # torch.save(model.quant_param_dict, 
    #             "/cv/hanjiarui/code/ViDiT-Q/examples/Wan2.1/quant_data/quant_params1.pth")
    save_checkpoint(transformer=model, rank=rank, output_dir="/cv/hanjiarui/code/ViDiT-Q/examples/Wan2.1/quant_data")

    logging.info(f'Quant params saved.')


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
