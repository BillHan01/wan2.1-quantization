from opensora.models.stdit.stdit3 import STDiT3Config
from models.quant_opensora import QuantOpenSora
from omegaconf import OmegaConf
import os
import time
from pprint import pformat

import colossalai
import torch
import torch.distributed as dist
import torch.nn as nn
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
from qdiff.base.quant_layer import QuantizedLinear

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
    ptq_config_file = cfg.get("ptq_config", None)
    quant_config = OmegaConf.load(ptq_config_file)
    
    '''
    INFO: modify the quant config to skip all quantization
    use the quant_model only for the apply_hook funcs
    '''
    quant_config.weight = None
    quant_config.act = None

    input_size = (num_frames, *image_size)
    latent_size = vae.get_latent_size(input_size)
    model = (
        build_module(
                cfg.model,
                MODELS,
                input_size=latent_size,
                in_channels=vae.out_channels,
                caption_channels=text_encoder.output_dim if not precompute_text_embeds else 4096,  # DIRTY FIX
                model_max_length=text_encoder.model_max_length if not precompute_text_embeds else 300,
                enable_sequence_parallelism=enable_sequence_parallelism,
            )
            .to(device, dtype)
            .eval()
    )  
    if not precompute_text_embeds:
        text_encoder.y_embedder = model.y_embedder  # HACK: for classifier-free guidance

    # == build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)
    
    '''
    INFO: the quant inference, however, no quantizer is applied, so no loading the quant_params
    '''
    # quant_param_ckpt = torch.load(os.path.join(cfg.save_dir,"./quant_params.pth"), weights_only=True)
    # model.load_quant_param_dict(quant_param_ckpt)
    # model.set_init_done()

    logger.info(str(model))
    
    
    # ------------ Add the Hooks for Calib Data --------------
    '''
    INFO: add the hook for hooking the activations, specify which layers to add. 
    Also, for Attention and Cross Attention, the hooks are specified during its init, 
    and could be accessed directly through tthe module. No need to add through `apply_func_to_submodule`
    '''
    kwargs = {
        'hook_cls': SaveActivationHook,
    }
    hook_d = apply_func_to_submodules(model,
                            class_type=nn.Linear,  # add hook to all objects of this cls
                            function=add_hook_to_module_,
                            return_d={},
                            **kwargs
                            )
  
    # ======================================================
    # inference
    # ======================================================
    # == load prompts ==
    prompts = cfg.get("prompt", None)
    start_idx = cfg.get("start_index", 0)
    if prompts is None:
        if cfg.get("prompt_path", None) is not None:
            prompts = load_prompts(cfg.prompt_path, start_idx, cfg.get("end_index", None))
        else:
            prompts = [cfg.get("prompt_generator", "")] * 1_000_000  # endless loop
    
    # == prepare reference ==
    reference_path = cfg.get("reference_path", [""] * len(prompts))
    mask_strategy = cfg.get("mask_strategy", [""] * len(prompts))
    assert len(reference_path) == len(prompts), "Length of reference must be the same as prompts"
    assert len(mask_strategy) == len(prompts), "Length of mask_strategy must be the same as prompts"

    # == prepare arguments ==
    fps = cfg.fps
    save_fps = cfg.get("save_fps", fps // cfg.get("frame_interval", 1))
    multi_resolution = cfg.get("multi_resolution", None)
    batch_size = cfg.get("batch_size", 1)
    num_sample = cfg.get("num_sample", 1)
    loop = cfg.get("loop", 1)
    condition_frame_length = cfg.get("condition_frame_length", 5)
    condition_frame_edit = cfg.get("condition_frame_edit", 0.0)
    align = cfg.get("align", None)

    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)
    sample_name = cfg.get("sample_name", None)
    prompt_as_path = cfg.get("prompt_as_path", False)
    

        # == Iter over all samples ==
    for i in progress_wrap(range(0, len(prompts), batch_size)):
        # == prepare batch prompts ==
        batch_prompts = prompts[i : i + batch_size]
        ms = mask_strategy[i : i + batch_size]
        refs = reference_path[i : i + batch_size]

        # == get json from prompts ==
        batch_prompts, refs, ms = extract_json_from_prompts(batch_prompts, refs, ms)
        original_batch_prompts = batch_prompts

        # == get reference for condition ==
        refs = collect_references_batch(refs, vae, image_size)

        # == multi-resolution info ==
        model_args = prepare_multi_resolution_info(
            multi_resolution, len(batch_prompts), image_size, num_frames, fps, device, dtype
        )

        # == Iter over number of sampling for one prompt ==
        for k in range(num_sample):
            # == prepare save paths ==
            save_paths = [
                get_save_path_name(
                    save_dir,
                    sample_name=sample_name,
                    sample_idx=start_idx + idx,
                    prompt=original_batch_prompts[idx],
                    prompt_as_path=prompt_as_path,
                    num_sample=num_sample,
                    k=k,
                )
                for idx in range(len(batch_prompts))
            ]

            # NOTE: Skip if the sample already exists
            # This is useful for resuming sampling VBench
            if prompt_as_path and all_exists(save_paths):
                continue

            # == process prompts step by step ==
            # 0. split prompt
            # each element in the list is [prompt_segment_list, loop_idx_list]
            batched_prompt_segment_list = []
            batched_loop_idx_list = []
            for prompt in batch_prompts:
                prompt_segment_list, loop_idx_list = split_prompt(prompt)
                batched_prompt_segment_list.append(prompt_segment_list)
                batched_loop_idx_list.append(loop_idx_list)

            # 1. refine prompt by openai
            if cfg.get("llm_refine", False):
                # only call openai API when
                # 1. seq parallel is not enabled
                # 2. seq parallel is enabled and the process is rank 0
                if not enable_sequence_parallelism or (enable_sequence_parallelism and is_main_process()):
                    for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                        batched_prompt_segment_list[idx] = refine_prompts_by_openai(prompt_segment_list)

                # sync the prompt if using seq parallel
                if enable_sequence_parallelism:
                    coordinator.block_all()
                    prompt_segment_length = [
                        len(prompt_segment_list) for prompt_segment_list in batched_prompt_segment_list
                    ]

                    # flatten the prompt segment list
                    batched_prompt_segment_list = [
                        prompt_segment
                        for prompt_segment_list in batched_prompt_segment_list
                        for prompt_segment in prompt_segment_list
                    ]

                    # create a list of size equal to world size
                    broadcast_obj_list = [batched_prompt_segment_list] * coordinator.world_size
                    dist.broadcast_object_list(broadcast_obj_list, 0)

                    # recover the prompt list
                    batched_prompt_segment_list = []
                    segment_start_idx = 0
                    all_prompts = broadcast_obj_list[0]
                    for num_segment in prompt_segment_length:
                        batched_prompt_segment_list.append(
                            all_prompts[segment_start_idx : segment_start_idx + num_segment]
                        )
                        segment_start_idx += num_segment

            # 2. append score
            for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                batched_prompt_segment_list[idx] = append_score_to_prompts(
                    prompt_segment_list,
                    aes=cfg.get("aes", None),
                    flow=cfg.get("flow", None),
                    camera_motion=cfg.get("camera_motion", None),
                )

            # 3. clean prompt with T5
            for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                batched_prompt_segment_list[idx] = [text_preprocessing(prompt) for prompt in prompt_segment_list]

            # 4. merge to obtain the final prompt
            batch_prompts = []
            for prompt_segment_list, loop_idx_list in zip(batched_prompt_segment_list, batched_loop_idx_list):
                batch_prompts.append(merge_prompt(prompt_segment_list, loop_idx_list))

            # == Iter over loop generation ==
            video_clips = []
            for loop_i in range(loop):
                # == get prompt for loop i ==
                batch_prompts_loop = extract_prompts_loop(batch_prompts, loop_i)

                # == add condition frames for loop ==
                if loop_i > 0:
                    refs, ms = append_generated(
                        vae, video_clips[-1], refs, ms, loop_i, condition_frame_length, condition_frame_edit
                    )

                # == sampling ==
                seed_everything(cfg.get("seed", 1024))  # DEBUG: somehow even if we have already set the seed before, we need to reinit here, maybe the seed changes within some imported files?
                z = torch.randn(len(batch_prompts), vae.out_channels, *latent_size, device=device, dtype=dtype)
                masks = apply_mask_strategy(z, refs, ms, loop_i, align=align)
                samples = scheduler.sample(
                    model,
                    text_encoder if not precompute_text_embeds else None,
                    z=z,
                    prompts=batch_prompts_loop,
                    device=device,
                    additional_args=model_args,
                    progress=verbose >= 2,
                    mask=masks,
                    precompute_text_embeds=precompute_text_embeds,
                )
                samples = vae.decode(samples.to(dtype), num_frames=num_frames)
                video_clips.append(samples)

            # == save samples ==
            # if is_main_process():
            #     for idx, batch_prompt in enumerate(batch_prompts):
            #         if verbose >= 2:
            #             logger.info("Prompt: %s", batch_prompt)
            #         save_path = save_paths[idx]
            #         video = [video_clips[i][idx] for i in range(loop)]
            #         for i in range(1, loop):
            #             video[i] = video[i][:, dframe_to_frame(condition_frame_length) :]
            #         video = torch.cat(video, dim=1)
            #         save_path = save_sample(
            #             video,
            #             fps=save_fps,
            #             save_path=save_path,
            #             verbose=verbose >= 2,
            #         )
            #         if save_path.endswith(".mp4") and cfg.get("watermark", False):
            #             time.sleep(1)  # prevent loading previous generated video
            #             add_watermark(save_path)
        start_idx += len(batch_prompts)
    logger.info("Inference finished.")
    logger.info("Saved %s samples to %s", start_idx, save_dir)
    
    # --------- Unpack the hooked results and save  -------------
    save_d = {}
    for k,v in hook_d.items():
        save_d[k] = torch.stack(v.outputs, dim=0)  # [N_timestep, C]
        logger.info(f'layer_name: {k}, hook_input_shape: {v.outputs[0].shape}')
        v.hook_handle.remove()

    torch.save(save_d, quant_config.calib_data.save_path)

if __name__ == "__main__":
    main()
