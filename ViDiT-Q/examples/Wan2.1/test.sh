torchrun --nproc_per_node=8 generate.py --task t2v-14B --sample_steps 30 --size 480*832 --ckpt_dir /vepfs-zulution/models/Wan2.1-T2V-14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "grance of trees, giving people a feeling of peace." --frame_num 81

# single GPU:
CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 fp_generate.py --task t2v-1.3B --sample_steps 30 --ckpt_dir /cv/hanjiarui/code/wan_new/Wan2.1/Wan2.1-T2V-1.3B --frame_num 81 --base_seed 42

CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 get_calib_data_wanx.py --task t2v-1.3B --sample_steps 30 --size 832*480 --ckpt_dir /cv/hanjiarui/code/wan_new/Wan2.1/Wan2.1-T2V-1.3B --frame_num 81 --base_seed 42

torchrun --nproc_per_node=1 ptq_wanx.py --task t2v-1.3B --sample_steps 30 --size 832*480 --ckpt_dir /cv/hanjiarui/code/wan_new/Wan2.1/Wan2.1-T2V-1.3B --prompt "grance of trees, giving people a feeling of peace." --frame_num 81 --quant_config "/cv/hanjiarui/code/ViDiT-Q/examples/Wan2.1/quant_configs/config.yaml" --base_seed 42

CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 ptq_wanx.py --task t2v-1.3B --sample_steps 3 --size 832*480 --ckpt_dir /cv/hanjiarui/code/wan_new/Wan2.1/Wan2.1-T2V-1.3B --frame_num 81 --quant_config "/cv/hanjiarui/code/ViDiT-Q/examples/Wan2.1/quant_configs/config.yaml" --base_seed 42

CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 quant_generate.py --task t2v-1.3B --sample_steps 30  --base_seed 42 \
 --ckpt_dir /cv/hanjiarui/code/wan_new/Wan2.1/Wan2.1-T2V-1.3B --size 832*480 \
 --frame_num 81 --quant_config "/cv/hanjiarui/code/ViDiT-Q/examples/Wan2.1/quant_configs/config.yaml"

# multi GPU:
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --nproc_per_node=6 fp_generate.py --task t2v-14B --sample_steps 30 --size 1280*720 --ckpt_dir /cv/hanjiarui/models/Wan2.1-T2V-14B --dit_fsdp --t5_fsdp --ulysses_size 6  --frame_num 81 --base_seed 42

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node=7 get_calib_data_wanx.py --task t2v-14B --sample_steps 30 --size 1280*720 --ckpt_dir /cv/hanjiarui/models/Wan2.1-T2V-14B  --frame_num 81 --base_seed 42  --t5_fsdp  --ulysses_size 7  --dit_fsdp

CUDA_VISIBLE_DEVICES=1,2,3,6 torchrun --nproc_per_node=4 get_calib_data_wanx.py --task t2v-14B --sample_steps 3 --size 832*480 --ckpt_dir /cv/hanjiarui/models/Wan2.1-T2V-14B  --frame_num 81 --base_seed 42  --t5_fsdp  --ulysses_size 4  

# tmp test at 1.3B
CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node=4 fp_generate.py --task t2v-1.3B --sample_steps 30 --size 832*480 --ckpt_dir /cv/hanjiarui/code/wan_new/Wan2.1/Wan2.1-T2V-1.3B  --t5_fsdp --ulysses_size 4  --frame_num 81 --base_seed 42 --dit_fsdp

CUDA_VISIBLE_DEVICES=1,3,6,7 torchrun --nproc_per_node=4 get_calib_data_wanx.py --task t2v-1.3B --sample_steps 30 --size 832*480 --ckpt_dir /cv/hanjiarui/code/wan_new/Wan2.1/Wan2.1-T2V-1.3B  --frame_num 81 --base_seed 42  --t5_fsdp  --ulysses_size 4  --dit_fsdp
CUDA_VISIBLE_DEVICES=1,3,6,7 torchrun --nproc_per_node=4 ptq_wanx.py --task t2v-1.3B --sample_steps 3 --size 832*480 --ckpt_dir /cv/hanjiarui/code/wan_new/Wan2.1/Wan2.1-T2V-1.3B  --base_seed 42 --quant_config "/cv/hanjiarui/code/ViDiT-Q/examples/Wan2.1/quant_configs/config.yaml"  --t5_fsdp  --ulysses_size 4 --dit_fsdp
CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node=4 quant_generate.py --task t2v-1.3B --sample_steps 30    \
 --ckpt_dir /cv/hanjiarui/code/wan_new/Wan2.1/Wan2.1-T2V-1.3B --size 832*480  --frame_num 81 --base_seed 42 \
 --quant_config "/cv/hanjiarui/code/ViDiT-Q/examples/Wan2.1/quant_configs/config.yaml"  --t5_fsdp --ulysses_size 4  --dit_fsdp

# single
CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 fp_generate.py --task t2v-1.3B --sample_steps 1 --size 832*480 --ckpt_dir /cv/hanjiarui/code/wan_new/Wan2.1/Wan2.1-T2V-1.3B  --frame_num 81 --base_seed 42 

CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 quant_generate.py --task t2v-1.3B --sample_steps 1    \
 --ckpt_dir /cv/hanjiarui/code/wan_new/Wan2.1/Wan2.1-T2V-1.3B --size 832*480  --frame_num 81 --base_seed 42 \
 --quant_config "/cv/hanjiarui/code/ViDiT-Q/examples/Wan2.1/quant_configs/config.yaml"  

CUDA_VISIBLE_DEVICES=1,2,3,6 torchrun --nproc_per_node=4 ptq_wanx.py --task t2v-14B --sample_steps 30 --size 1280*720 --ckpt_dir /cv/hanjiarui/models/Wan2.1-T2V-14B  --base_seed 42 --quant_config "/cv/hanjiarui/code/ViDiT-Q/examples/Wan2.1/quant_configs/config.yaml"  --t5_fsdp  --ulysses_size 4 --dit_fsdp

torchrun --nproc_per_node=8 quant_generate.py --task t2v-14B --sample_steps 30  --dit_fsdp --t5_fsdp --ulysses_size 8  \
 --ckpt_dir /cv/hanjiarui/models/Wan2.1-T2V-14B --size 1280*720  --frame_num 81 --base_seed 42 \
 --quant_config "/cv/hanjiarui/code/ViDiT-Q/examples/Wan2.1/quant_configs/config.yaml"
