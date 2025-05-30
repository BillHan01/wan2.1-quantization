import argparse
import json
import os
import numpy as np
import torch
from tqdm import tqdm
import random
import sys
# sys.path.append("/mnt/public/video_quant/chenxianying/ditq_eval/diffuser-dev-eval")
sys.path.append("/mnt/public/diffusion_quant/zhaotianchen/project/viditq/clean/eval/image")
import evaluation.metrics as RM
from evaluation.fid_score import calculate_fid_given_paths
import matplotlib.pyplot as plt
import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CocoDataset(Dataset):
    def __init__(self, args):
        self.model_img_dirpath = args.img_dir
        self.prompts_ids = []
        if args.prompts_path[-4:] == "json":
            with open(args.prompts_path, 'r', encoding = 'utf-8') as f:
                for i, j in enumerate(f.readlines()):
                    j = json.loads(j)
                    self.prompts_ids.append(j)
        elif args.prompts_path[-3:] == "txt":
            with open(args.prompts_path, 'r') as file:
                for line in file:
                    self.prompts_ids.append(line.strip())

    def __getitem__(self,index):
        if args.prompts_path[-4:] == "json":
            id = self.prompts_ids[index]['id']
            prompt = self.prompts_ids[index]['caption']
        elif args.prompts_path[-3:] == "txt":
            id = int(index)
            prompt = self.prompts_ids[index]
        # image_path = os.path.join(self.model_img_dirpath, f"{id}.png")
        image_path = os.path.join(self.model_img_dirpath, f"output_{index}.jpg")
        return id, prompt, image_path

    def __len__(self):
        return len(self.prompts_ids)


def test_image_score(image_paths, prompt, metric, device, **kwargs):
    if metric.lower() == "fid":
        rewards = calculate_fid_given_paths(
            [image_paths, kwargs["source_images"]],
            batch_size=kwargs["batch_size"],
            device=device,
            load_act=[None, kwargs["load_act"]],
            save_act=[None, kwargs["save_act"]],
            img_size=kwargs["img_size"]
        )
        return None, rewards    
    model = RM.load_score(name=metric, device=device)
    # indices, rewards = model.score(prompt, image_paths)
    with torch.no_grad():
        indices, rewards = model.score(prompt, image_paths)
    return indices, rewards
    # return rewards

# def test_dir(image_dir, prompts, metric, device):


def test_dataset_score(args):
    if torch.cuda.is_available():
        device = torch.device(
            f"cuda:{args.gpu_id}" if args.gpu_id is not None else "cuda"
        )
    else:
        device = torch.device("cpu")

    model = RM.load_score(name=args.metric, device=device)
    dataset = CocoDataset(args)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False) 
 
    with torch.no_grad():
        result11 = []
        score_list = []
        for i, (ids, prompts, image_paths) in enumerate(dataloader): 
                id, prompt, image_path = ids[0], prompts[0], image_paths[0]
                id = int(id)
                if os.path.exists(image_path):
                    indices, rewards = model.score(prompt, image_path)
                    # print(rewards)
                    score_list.append(rewards)
                    result11.append((id,rewards))
                else:
                    continue

        # 打开文件并将字符串写入
        path = f'result_{args.mode}/'
        os.makedirs(path, exist_ok=True)
        with open(f'{path}/result-score.json',"w",) as f:
            json.dump(dict(result11), f, indent=4, ensure_ascii=False)
        avg_val = sum(dict(result11).values()) / len(result11)
        if args.log_file:
            with open(args.log_file, 'a') as file:
                # Write the new line to the file
                file.write(f"{args.img_dir} {args.mode}: {avg_val}" + '\n')

        print(f"{args.img_dir} {args.mode}: \n {avg_val}" + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        default="A very cute white dog standing by a table with food",
        type=str,
        help="content of prompt",
    )
    parser.add_argument(
        "--image_paths",
        default="coco_caption_imgs/A very cute white dog standing by a table with food/",
        type=str,
        help="file (e.g. png, jpg) or folder (multiple images) of image path",
    )
    parser.add_argument(
        "--metric",
        default="HPS",
        type=str,
        help="ImageReward, HPS, Aesthetic, CLIP, BLIP, PickScore",
    )
    parser.add_argument(
        "--gpu_id",
        default='1',
        type=str,
        help="GPU ID(s) to use for CUDA.",
    )
    parser.add_argument(
        "--mode",
        default= 'set',
        type=str,
        help="set or one (to evaluate dataset or one prompt)",
    )
    parser.add_argument(
        "--img_dir",
        default= '/homez/zhaolin-local/generated_images/allimageitem/dreamlike-photoreal/DDIM10/fp/seed0/',
        type=str,
        help="path of images generated by the prompts in the dataset",
    )
    parser.add_argument(
        "--prompts_path",
        default= 'Coco_caption/prompt-image.json',
        type=str,
        help="prompts path",
    )
    parser.add_argument(
        "--log_file",
        default= None,
        type=str,
        help="log file",
    )

    args = parser.parse_args()


    if args.mode == 'one':
        if torch.cuda.is_available():
            device = torch.device(
                f"cuda:{args.gpu_id}" if args.gpu_id is not None else "cuda"
            )
        else:
            device = torch.device("cpu")
        indices, rewards = test_image_score(args.image_paths, args.prompt, args.metric, device)
        if os.path.isdir(args.image_paths):
            print("ranking: ", indices)
            print("score: ", rewards)
        else:
            print("score: ", rewards)
    elif args.mode == 'set':
        test_dataset_score(args)
