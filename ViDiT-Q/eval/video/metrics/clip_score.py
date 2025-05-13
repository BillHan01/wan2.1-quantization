import os
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
import time
import logging
# import wandb
from tqdm import tqdm
import argparse
import torchvision.transforms as transforms
from torchvision.transforms import Resize
from torchvision.utils import save_image
import requests

def extract_number_customize(filename):
    try:
        # 提取数字部分并转换为整数
        return int(filename.split('_')[-1].split('.')[0])
    except (ValueError, IndexError):
        # 如果不符合格式，返回一个默认值（例如 -1）
        return -1

def calculate_clip_score(video_path, text, model, tokenizer):
    
    print("cur_text:",text)
    
    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Extract frames from the video 
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(frame,(224,224))  # Resize the frame to match the expected input size
        frames.append(resized_frame)

    # Convert numpy arrays to tensors, change dtype to float, and resize frames
    tensor_frames = [torch.from_numpy(frame).permute(2, 0, 1).float() for frame in frames]

    # Initialize an empty tensor to store the concatenated features
    concatenated_features = torch.tensor([], device=device)

    # Generate embeddings for each frame and concatenate the features
    with torch.no_grad():
        for frame in tensor_frames:
            frame_input = frame.unsqueeze(0).to(device)  # Add batch dimension and move the frame to the device
            frame_features = model.get_image_features(frame_input)
            concatenated_features = torch.cat((concatenated_features, frame_features), dim=0)

    # Tokenize the text
    text_tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=77)

    # Convert the tokenized text to a tensor and move it to the device
    text_input = text_tokens["input_ids"].to(device)

    # Generate text embeddings
    with torch.no_grad():
        text_features = model.get_text_features(text_input)

    # Calculate the cosine similarity scores
    concatenated_features = concatenated_features / concatenated_features.norm(p=2, dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    clip_score_frames = concatenated_features @ text_features.T
    # Calculate the average CLIP score across all frames, reflects temporal consistency 
    clip_score_frames_avg = clip_score_frames.mean().item()

    return clip_score_frames_avg

def calculate_clip_temp_score(video_path, model):
    # Load the video
    cap = cv2.VideoCapture(video_path)
    to_tensor = transforms.ToTensor()
    # Extract frames from the video 
    frames = []
    SD_images = []
    resize = transforms.Resize([224,224])
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # resized_frame = cv2.resize(frame,(224,224))  # Resize the frame to match the expected input size
        frames.append(frame)
    
    tensor_frames = torch.stack([resize(torch.from_numpy(frame).permute(2, 0, 1).float()) for frame in frames])

    # tensor_frames = [extracted_frames[i] for i in range(extracted_frames.size()[0])]
    concatenated_frame_features = []

    # Generate embeddings for each frame and concatenate the features
    with torch.no_grad():  
        for frame in tensor_frames: # Too many frames in a video, must split before CLIP embedding, limited by the memory
            frame_input = frame.unsqueeze(0).to(device)  # Add batch dimension and move the frame to the device
            frame_feature = model.get_image_features(frame_input)
            concatenated_frame_features.append(frame_feature)

    concatenated_frame_features = torch.cat(concatenated_frame_features, dim=0)

    # Calculate the similarity scores
    clip_temp_score = []
    concatenated_frame_features = concatenated_frame_features / concatenated_frame_features.norm(p=2, dim=-1, keepdim=True)
    # ipdb.set_trace()

    for i in range(concatenated_frame_features.size()[0]-1):
        clip_temp_score.append(concatenated_frame_features[i].unsqueeze(0) @ concatenated_frame_features[i+1].unsqueeze(0).T)
    clip_temp_score=torch.cat(clip_temp_score, dim=0)
    # Calculate the average CLIP score across all frames, reflects temporal consistency 
    clip_temp_score_avg = clip_temp_score.mean().item()

    return clip_temp_score_avg

def sort_key(item):
    return int(item.split('_')[1].split('.')[0])

def extract_number(filename):
    return int(filename.split('_')[1].split('.')[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_videos", type=str, default='', help="Specify the path of generated videos")
    parser.add_argument("--dir_prompts", type=str, default='', help="Specify the path of prompts")
    parser.add_argument("--dir_results", type=str, default='', help="Specify the path of results")
    parser.add_argument("--metric", type=str, default='celebrity_id_score', help="Specify the metric to be used")
    args = parser.parse_args()

    dir_videos = args.dir_videos
    metric = args.metric
    dir_prompts = args.dir_prompts
    dir_results = args.dir_results
   
    #video_list = sorted(os.listdir(dir_videos), key=sort_key)
    file_list = os.listdir(dir_videos)
    # 过滤掉不符合格式的文件
    filtered_file_list = [f for f in file_list if extract_number_customize(f) != -1] 
    video_list = sorted(filtered_file_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    video_paths = [os.path.join(dir_videos, x) for x in video_list]
    #print(video_paths)
    #prompt_paths = [os.path.join(dir_prompts, os.path.splitext(os.path.basename(x))[0]+'.txt') for x in video_paths]

     # Create the directory if it doesn't exist
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    #os.makedirs(dir_results+f"/{timestamp}")
    # Set up logging
    log_file_path = dir_results
    #filename = f"/{timestamp}/{metric}_record.txt"
    filename = f"/{metric}_record.txt"
    log_file = log_file_path + filename
    # Set up logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # File handler for writing logs to a file
    file_handler = logging.FileHandler(filename=log_file )
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)
    # Stream handler for displaying logs in the terminal
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(stream_handler)


    # Load pretrained models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    # Calculate SD scores for all video-text pairs
    scores = []
    
    test_num = 10
    test_num = len(video_paths)
    
    with open(dir_prompts, 'r', encoding='utf-8') as file:
        texts = [line.strip() for line in file.readlines()]
    
    count = 0
    for i in tqdm(range(len(video_paths))):
        video_path = video_paths[i]
        #prompt_path = prompt_paths[i]
        text = texts[i]
        if count == test_num:
            break
        else:
            # ipdb.set_trace()
            if metric == 'clip_score':
                score = calculate_clip_score(video_path, text, clip_model, clip_tokenizer)
            elif metric == 'clip_temp_score':
                score = calculate_clip_temp_score(video_path,clip_model)
            count+=1
            scores.append(score)
            average_score = sum(scores) / len(scores)
            # count+=1
            logging.info(f"Vid: {os.path.basename(video_path)},  Current {metric}: {score}, Current avg. {metric}: {average_score},  ")

    # Calculate the average SD score across all video-text pairs
    logging.info(f"Final average {metric}: {average_score}, Total videos: {len(scores)}")
