# Video Metrics Evaluation

## Introduction

This folder shows how we evaluate the generated videos. The evaluation contains two settings:    

(1) Benchmark suite: We evaluate the quantized model on VBench to provide comprehensive results. We select 8 major dimensions from Vbench.  

(2)Multiaspects metrics: we select the following metrics from different perspectives:

- *CLIPSIM* and *CLIP-Temp* to measure the text-video alignment and temporal semantic consistency, and DOVER's video quality
- (*VQA*) metrics to evaluate the generation quality from aesthetic (high-level) and technical
(low-level) perspectives
- *Flow-score* and *Temporal Flickering* are used for evaluating the temporal
consistency.
- *FPFVD* to measure the difference between the videos generated with FP16 and the videos generated with quantized model.

## Usage

### VBench
For VBench evalutation, you can clone the [VBench](https://github.com/Vchitect/VBench) codebase and follow the `VBench/README.md` to evaluate the videos.

### Multi Aspects metrics

#### 1. Env Setup
We recommend using conda for enviornment management.
```bash
# create a virtual env and activate
conda create -n metrics python==3.10 
conda activate metrics

# install torch
pip install torch torchvision torchaudio

#install the requirements
pip install -r requirements.txt
```
#### 2. Download checkpoints

When calculating the FPFVD, you need the `i3d_torchscript.pt` which can be downloaded on https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1.

Other checkpoints can be downloaded automatically. If that doesn't work, you can try to download manually on huggingface.

#### 3. Evaluation
You need to modify 3 args in `metrics/simplebench_comp_101.sh`. 
- `dir_videos`: the path of the floder with generated videos.
- `dir_prompts`: the prompts used for generation.
- `dir_ref_vid_fp`: the path of the floder with FP videos.

After modify the parameters, you can run the `metrics/simplebench_comp_101.sh` and you will get the metrics.