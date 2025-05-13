# Wan2.1-quantization

## 🖥️ Runtime Environment

This project was developed and tested on the following hardware and software environment:

```text
OS           : Ubuntu 20.04.6 LTS (focal)  
Python       : 3.10.0  
PyTorch      : 2.6.0+cu124  
CUDA (Torch) : 12.4  
cuDNN        : 9.1.0  
CUDA Toolkit : 12.8 (V12.8.93)  
GPU Model    : 8 × NVIDIA A800-SXM4-80GB  
```

## 🛠️ Setup Instructions

We recommend using Conda to create a reproducible environment:

```bash
# Step 1: Create a new conda environment
conda env create -f ViDiT-Q/environment.yml

# Step 2: Activate the environment
conda activate wan2.1-quant
```

The environment.yml defines all necessary dependencies.

## 🚀 How to Run
This project supports FP inference, PTQ calibration, quantization, and quantized inference for both 1.3B and 14B Wan 2.1 T2V models.

Make sure your environment is correctly set up with torchrun, and the necessary model checkpoints and quantization configs are prepared.

### 1. FP Inference (Full Precision)

#### 1.3B Model (Single GPU)
```bash
torchrun --nproc_per_node=1 fp_generate.py \
  --task t2v-1.3B \
  --sample_steps 1 \
  --size 832*480 \
  --ckpt_dir /path/to/Wan2.1-T2V-1.3B \
  --frame_num 81 \
  --base_seed 42
```
#### 14B Model (Multi-GPU, 8x)
```bash
torchrun --nproc_per_node=8 fp_generate.py \
  --task t2v-14B \
  --sample_steps 30 \
  --size 1280*720 \
  --ckpt_dir /path/to/Wan2.1-T2V-14B \
  --frame_num 81 \
  --base_seed 42 \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size 8
```

### 2. Calibration (for PTQ)
#### 1.3B Model
```bash
torchrun --nproc_per_node=1 get_calib_data_wanx.py \
  --task t2v-1.3B \
  --sample_steps 30 \
  --size 832*480 \
  --ckpt_dir /path/to/Wan2.1-T2V-1.3B \
  --frame_num 81 \
  --base_seed 42
```
#### 14B Model
```bash
torchrun --nproc_per_node=8 get_calib_data_wanx.py \
  --task t2v-14B \
  --sample_steps 30 \
  --size 1280*720 \
  --ckpt_dir /path/to/Wan2.1-T2V-14B \
  --frame_num 81 \
  --base_seed 42 \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size 8
```

### 3. PTQ Execution (Post-Training Quantization)
#### 1.3B Model
```bash
torchrun --nproc_per_node=4 ptq_wanx.py \
  --task t2v-1.3B \
  --sample_steps 3 \
  --size 832*480 \
  --ckpt_dir /path/to/Wan2.1-T2V-1.3B \
  --base_seed 42 \
  --quant_config /path/to/quant_configs/config.yaml
```
#### 14B Model
```bash
torchrun --nproc_per_node=8 ptq_wanx.py \
  --task t2v-14B \
  --sample_steps 30 \
  --size 1280*720 \
  --ckpt_dir /path/to/Wan2.1-T2V-14B \
  --frame_num 81 \
  --base_seed 42 \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size 8 \
  --quant_config /path/to/quant_configs/config.yaml
```

### 4. Quantized Inference
#### 1.3B Model
```bash
torchrun --nproc_per_node=1 quant_generate.py \
  --task t2v-1.3B \
  --sample_steps 1 \
  --size 832*480 \
  --ckpt_dir /path/to/Wan2.1-T2V-1.3B \
  --frame_num 81 \
  --base_seed 42 \
  --quant_config /path/to/quant_configs/config.yaml
```
#### 14B Model
```bash
torchrun --nproc_per_node=8 quant_generate.py \
  --task t2v-14B \
  --sample_steps 30 \
  --size 1280*720 \
  --ckpt_dir /path/to/Wan2.1-T2V-14B \
  --frame_num 81 \
  --base_seed 42 \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size 8 \
  --quant_config /path/to/quant_configs/config.yaml
```
> Notes:
> - Replace /path/to/... with your actual model and config paths.
> - --dit_fsdp, --t5_fsdp, and --ulysses_size are required for large-scale multi-GPU inference.
> - Use --sample_steps to control generation speed and quality trade-offs.

## 🔗 Related Repositories
> **Wan2.1 Original Repository:**
> https://github.com/Wan-Video/Wan2.1
> Official source code and pretrained checkpoints for the Wan2.1 Text-to-Video model.
>
> **ViDiT-Q Quantization Toolkit:**
> https://github.com/thu-nics/ViDiT-Q
> Post-training quantization framework for diffusion transformers, used as the backbone for this project’s quantization pipeline.