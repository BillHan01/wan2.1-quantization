# Image Metrics Evaluation

## Introduction
This folder shows how we evaluate the generated images. We choose：
- *Clipscore* for text-image alignment
- *ImageReward* for human preference. 
- *FPFID* to measure the difference between the images generated with FP16 and the images generated with quantized model.

## Env Setup

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

## How to use 

The `coco_1024.txt` is a selected subset of COCO annotations. 

### Eval FID
Modify the parameters and run **./evaluation/fid.sh**    
- fp_path: the path of fp images    
- base_dir: the path of images to be evaluated   
- log_file: the path of log file    
    
### Eval CLIP

Download the `ViT-L-14.pt` from this [Link](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt), and place it under the `~/.cache/metrics_models`.
Modify the parameters and run **./evaluation/test-score.sh**      
metric: "CLIP"   
- dir: the path of images to be evaluated   
- log_file: the path of log file  
- prompt_path: the path of prompts    

### Eval ImageReward

Download the `ImageReward.pt` and `med_config.json` on https://huggingface.co/THUDM/ImageReward/tree/main, and place it under the `~/.cache/metrics_models`.

Modify the parameters and run **./evaluation/test-score.sh**      
- metric: "ImageReward"   
- dir: the path of images to be evaluated   
- log_file: the path of log file  
- prompt_path: the path of prompts  
