from opensora.models.stdit.stdit3 import STDiT3Config
from models.quant_opensora import QuantOpenSora
from omegaconf import OmegaConf, ListConfig
import os
import time
from pprint import pformat

import colossalai
import torch
import torch.distributed as dist
from colossalai.cluster import DistCoordinator
from mmengine.runner import set_random_seed
from tqdm import tqdm

from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.datasets import save_sample
from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.models.text_encoder.t5 import text_preprocessing
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.inference_utils import (
    add_watermark,
    append_generated,
    append_score_to_prompts,
    apply_mask_strategy,
    collect_references_batch,
    dframe_to_frame,
    extract_json_from_prompts,
    extract_prompts_loop,
    get_save_path_name,
    load_prompts,
    merge_prompt,
    prepare_multi_resolution_info,
    refine_prompts_by_openai,
    split_prompt,
)
from opensora.utils.misc import all_exists, create_logger, is_distributed, is_main_process, to_torch_dtype
from qdiff.utils import apply_func_to_submodules, seed_everything


def main():
    torch.set_grad_enabled(False)
    # ======================================================
    # configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dtype = cfg.get("dtype", "fp32")
    assert cfg_dtype in ["fp16", "bf16", "fp32"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # == init distributed env ==
    if is_distributed():
        colossalai.launch_from_torch({})
        coordinator = DistCoordinator()
        enable_sequence_parallelism = coordinator.world_size > 1
        if enable_sequence_parallelism:
            set_sequence_parallel_group(dist.group.WORLD)
    else:
        coordinator = None
        enable_sequence_parallelism = False
    seed_everything(cfg.get("seed", 1024))
    # set_random_seed(seed=cfg.get("seed", 1024))
    
    # == bakup some files ==
    import shutil
    if os.path.exists(os.path.join(cfg.save_dir,'configs')):
        shutil.rmtree(os.path.join(cfg.save_dir,'configs'))
    shutil.copytree('./configs', os.path.join(cfg.save_dir,'configs'))

    # == init logger ==
    logger = create_logger()
    logger.info("Inference configuration:\n %s", pformat(cfg.to_dict()))
    verbose = cfg.get("verbose", 1)
    progress_wrap = tqdm if verbose == 1 else (lambda x: x)
    
    # INFO: precompute the text embeds to avoid loading the T5 repeatedly
    precompute_text_embeds = cfg.get("precompute_text_embeds", False)
    #assert precompute_text_embeds # DEBUG_ONLY

    # ======================================================
    # build model & load weights
    # ======================================================
    logger.info("Building models...")
    # == build text-encoder and vae ==
    if not precompute_text_embeds:
        text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()


    # == prepare video size ==
    image_size = cfg.get("image_size", None)
    if image_size is None:
        resolution = cfg.get("resolution", None)
        aspect_ratio = cfg.get("aspect_ratio", None)
        assert (
            resolution is not None and aspect_ratio is not None
        ), "resolution and aspect_ratio must be provided if image_size is not provided"
        image_size = get_image_size(resolution, aspect_ratio)
    num_frames = get_num_frames(cfg.num_frames)

    # == build diffusion model ==
    quant_config = cfg.get("ptq_config", None)
    quant_config = OmegaConf.load(quant_config)
    input_size = (num_frames, *image_size)
    latent_size = vae.get_latent_size(input_size)
    config = STDiT3Config(depth=28, 
                        hidden_size=1152, 
                        patch_size=(1, 2, 2), 
                        num_heads=16, 
                        qk_norm=True,
                        enable_flash_attn=True,
                        enable_layernorm_kernel=False,  # no apex included
                        input_size=latent_size,
                        in_channels=vae.out_channels,
                        caption_channels=text_encoder.output_dim if not precompute_text_embeds else 4096,
                        model_max_length=text_encoder.model_max_length if not precompute_text_embeds else 300,
                        enable_sequence_parallelism=enable_sequence_parallelism)
    # model_from_pretrained=os.path.join(cfg.model_path, "hpcai-tech/OpenSora-STDiT-v3")
    model=(QuantOpenSora(quant_config,config,"/cv/hanjiarui/models/OpenSora-STDiT-v3/model.safetensors").to(device, dtype).eval())  
    # model=(QuantOpenSora(quant_config,config,model_from_pretrained).to(device, dtype).eval())  
    if not precompute_text_embeds:
        text_encoder.y_embedder = model.y_embedder  # HACK: for classifier-free guidance
    if_mixed_precision = isinstance(quant_config.weight.n_bits, ListConfig) or isinstance(quant_config.act.n_bits, ListConfig)
    if if_mixed_precision:
        model.bitwidth_refactor()
    # == build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)
    
    '''
    INFO: The PTQ process:
    for simple PTQ with dynamic act quant: 
    the weight are quantized with quant_model initialization.
    the act quant params are calculated online. 
    '''
    
    # TODO: some variables (quant_config) are not replaced yet, havent test sq and quarot!
    
    def init_sq_channel_mask_(module, full_name, calib_data):
        assert isinstance(module, SQQuantizedLinear)
        act_mask = calib_data[full_name].max(dim=0)[0]  # [T, C], averaged over all timesteps
        zero_mask = act_mask < 1e-3
        act_mask = torch.where(zero_mask, torch.tensor(1e-3), act_mask)
        module.get_channel_mask(act_mask)  # set self.channel_mask
        module.update_quantized_weight_scaled()

    def init_rotation_matrix_(module, full_name):
        from qdiff.quarot.quarot_utils import random_hadamard_matrix, matmul_hadU_cuda
        assert isinstance(module, QuarotQuantizedLinear)
        module.get_rotation_matrix()
        module.update_quantized_weight_rotated()
    
    def init_rotation_and_channel_mask_(module, full_name, calib_data):
        assert isinstance(module, ViDiTQuantizedLinear)
        act_mask = calib_data[full_name].max(dim=0)[0]  # [T, C], averaged over all timesteps
        zero_mask = act_mask < 1e-3
        act_mask = torch.where(zero_mask, torch.tensor(1e-3), act_mask)
        module.get_channel_mask(act_mask)  # set self.channel_mask
        module.get_rotation_matrix()
        module.update_quantized_weight_rotated_and_scaled()

    '''
    INFO: the smooth_quant quantization.
    load act channel mask from the calib data
    '''
    if quant_config.get("smooth_quant",None) is not None:
        # INFO: the SQQuantizedLayer are initialized with the quant_layer_refactor_ in quant_dit.py
        from qdiff.smooth_quant.sq_quant_layer import SQQuantizedLinear

        assert quant_config.calib_data.save_path is not None
        calib_data = torch.load(quant_config.calib_data.save_path, weights_only=True)  # default wtih weights_only=True, will cause warning

        # get the channel mask, iter through all layers
        kwargs = {}
        apply_func_to_submodules(model,
                            class_type=SQQuantizedLinear,  # add hook to all objects of this cls
                            function=init_sq_channel_mask_,
                            calib_data = calib_data,
                            full_name='',
                            **kwargs
                            )

    '''
    INFO: the quarot quantization.
    init and apply the rotation matrix
    '''
    if quant_config.get("quarot",None) is not None:
        
        from qdiff.quarot.quarot_quant_layer import QuarotQuantizedLinear
        # get the rotation matrix, iter through all layers
        kwargs = {}
        apply_func_to_submodules(model,
                            class_type=QuarotQuantizedLinear,  # add hook to all objects of this cls
                            function=init_rotation_matrix_,
                            full_name='',
                            **kwargs
                            )
    '''
    INFO: combining both
    '''
    if quant_config.get("viditq",None) is not None:
        from qdiff.viditq.viditq_quant_layer import ViDiTQuantizedLinear
        
        assert quant_config.calib_data.save_path is not None
        calib_data = torch.load(quant_config.calib_data.save_path, weights_only=True)  # default wtih 
        kwargs = {}
        apply_func_to_submodules(model,
                            class_type=ViDiTQuantizedLinear,  # add hook to all objects of this cls
                            function=init_rotation_and_channel_mask_,
                            full_name='',
                            calib_data = calib_data,
                            **kwargs
                            )
        
    model.set_init_done()
    model.save_quant_param_dict()
    torch.save(model.quant_param_dict, os.path.join(cfg.save_dir, 'quant_params.pth'))
    logger.info(f'saved quant params into {cfg.save_dir}')

    layer = model.spatial_blocks[11]  
    weights = layer.state_dict()

    torch.save(weights, 'spatial_blocks.11.pth')

    

if __name__ == "__main__":
    main()
