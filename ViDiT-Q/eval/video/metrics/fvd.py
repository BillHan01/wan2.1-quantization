# ref: https://github.com/universome/fvd-comparison/blob/master/our_fvd.py

import numpy as np
import torch
from fvd_utils import open_url
import os
import random
import cv2
from typing import Tuple
import scipy
import argparse
import logging



def extract_number(filename):
    return int(filename.split('_')[1].split('.')[0])



def compute_fvd(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
    mu_gen, sigma_gen = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))

    return float(fid)


def compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    feats = feats.reshape(feats.shape[1], -1)
    mu = feats.mean(axis=0) # [d]
    sigma = np.cov(feats, rowvar=False) # [d, d]

    return mu, sigma

def compute_our_fvd(videos_fake: np.ndarray, 
                    videos_real: np.ndarray, 
                    batch_size=50,
                    num_videos=101,
                    device: str='cuda') -> float:
    print(videos_fake.shape)
    print(videos_real.shape)
    detector_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
    detector_kwargs = dict(rescale=False, resize=False, return_features=True) # Return raw features before the softmax layer.

    """
    with open_url(detector_url, verbose=False) as f:
        detector = torch.jit.load(f).eval().to(device)
    """
    detector = torch.jit.load('$YOUR_PATH/.cache/frechat/i3d_torchscript.pt',map_location=device).eval().to(device)
    
    

    videos_fake = torch.from_numpy(videos_fake).permute(0, 4, 1, 2, 3).to(device)
    videos_real = torch.from_numpy(videos_real).permute(0, 4, 1, 2, 3).to(device)
    
    #print(videos_fake)
    
    #videos_fake = torch.cuda.FloatTensor(videos_fake)
    #videos_real = torch.cuda.FloatTensor(videos_real)

    vshape = (-1,3,16,224,224)
    
    for i in range(0,num_videos,batch_size):
        if i==0:
            feats_fake = detector(videos_fake[i:i+batch_size].reshape(vshape), **detector_kwargs).cpu()
            feats_real = detector(videos_real[i:i+batch_size].reshape(vshape), **detector_kwargs).cpu()
        elif i + batch_size < num_videos:
            n_ff = detector(videos_fake[i:i+batch_size].reshape(vshape), **detector_kwargs).cpu()
            feats_fake = torch.cat((feats_fake, n_ff), axis=0)
            n_fr = detector(videos_real[i:i+batch_size].reshape(vshape), **detector_kwargs).cpu()
            feats_real = torch.cat((feats_real, n_fr), axis=0)
        else:
            n_ff = detector(videos_fake[i:num_videos].reshape(vshape), **detector_kwargs).cpu()
            feats_fake = torch.cat((feats_fake, n_ff), axis=0)
            n_fr = detector(videos_real[i:num_videos].reshape(vshape), **detector_kwargs).cpu()
            feats_real = torch.cat((feats_real, n_fr), axis=0)
        
        #print(i)
    
    feats_fake = feats_fake.detach().numpy()
    feats_real = feats_real.detach().numpy()

    return compute_fvd(feats_fake, feats_real)

def read_ucf101_simple(base_path='/share/public/video_quant/wanrui/datasets/ucf101/videos/UCF-101'):
    # 从每个class中随机抽出一个视频
    output_size = (224, 224)

    # 存储所有视频张量的列表
    video_tensors = []

    # 遍历每个文件夹
    dir_list = sorted(os.listdir(base_path))
    #print(dir_list)
    for folder in dir_list:
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            # 获取文件夹中的所有avi文件
            avi_files = [f for f in os.listdir(folder_path) if f.endswith('.avi')]
            if avi_files:
                # 随机选择一个avi文件
                random_avi_file = random.choice(avi_files)
                video_path = os.path.join(folder_path, random_avi_file)
                
                # 读取视频
                cap = cv2.VideoCapture(video_path)
                frames = []
                i = 0
                while cap.isOpened() and i < 16:
                    i += 1
                    ret, frame = cap.read()
                    frame = frame.astype(np.float32)
                    frame /= 255
                    if not ret:
                        break
                    # 截取中间的240*240
                    center_frame = frame[40:280, 40:280]
                    # resize到256*256
                    resized_frame = cv2.resize(center_frame, output_size)
                    frames.append(resized_frame)
                cap.release()
                
                # 将所有帧组合成一个张量
                video_tensor = np.stack(frames)
                video_tensors.append(video_tensor)

    # 将所有张量连接到一起
    final_tensor = np.concatenate(video_tensors, axis=0)
    final_tensor = final_tensor.reshape(-1,16,224,224,3)

    # 输出最终张量的形状
    #print("Final tensor shape:", final_tensor.shape)
    
    return final_tensor


def read_ucf101_full(base_path='/share/public/video_quant/wanrui/datasets/ucf101/videos/UCF-101',batch_start=0,batch_end=101):
    # 从每个class中随机抽出一个视频
    output_size = (224, 224)

    # 存储所有视频张量的列表
    video_tensors = []

    # 遍历每个文件夹
    dir_list = sorted(os.listdir(base_path))
    #print(dir_list)
    no = 0
    for num in range(batch_start,batch_end):
        folder = dir_list[num]
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            # 获取文件夹中的所有avi文件
            avi_files = [f for f in os.listdir(folder_path) if f.endswith('.avi')]
            if avi_files:
                # 随机选择一个avi文件
                if no<28:
                    num_sample = 21
                else:
                    num_sample = 20
                random_avi_files = random.sample(avi_files, num_sample)
                
                for random_avi_file in random_avi_files:
                    video_path = os.path.join(folder_path, random_avi_file)
                    # 读取视频
                    cap = cv2.VideoCapture(video_path)
                    frames = []
                    i = 0
                    while cap.isOpened() and i < 16:
                        i += 1
                        ret, frame = cap.read()
                        frame = frame.astype(np.float32)
                        frame /= 255
                        if not ret:
                            break
                        # 截取中间的240*240
                        center_frame = frame[40:280, 40:280]
                        # resize到256*256
                        resized_frame = cv2.resize(center_frame, output_size)
                        frames.append(resized_frame)
                    cap.release()
                    
                    # 将所有帧组合成一个张量
                    video_tensor = np.stack(frames)
                    video_tensors.append(video_tensor)

    # 将所有张量连接到一起
    final_tensor = np.concatenate(video_tensors, axis=0)
    final_tensor = final_tensor.reshape(-1,16,224,224,3)

    # 输出最终张量的形状
    #print("Final tensor shape:", final_tensor.shape)
    
    return final_tensor

def extract_number(filename):
    try:
        # 提取数字部分并转换为整数
        return int(filename.split('_')[-1].split('.')[0])
    except (ValueError, IndexError):
        # 如果不符合格式，返回一个默认值（例如 -1）
        return -1
 
def read_generated(base_path,output_size=(224, 224)):
    # 存储所有视频张量的列表
    video_tensors = []

    # 遍历每个文件
    file_list = os.listdir(base_path)
    # 过滤掉不符合格式的文件
    filtered_file_list = [f for f in file_list if extract_number(f) != -1] 
    video_list = sorted(filtered_file_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    video_paths = [os.path.join(base_path, x) for x in video_list]
    #print(video_paths)
    for file_name in video_paths:
        file_path = os.path.join(base_path, file_name)
        nf = 0
        if file_name.endswith('.mp4'):
            # 读取视频
            cap = cv2.VideoCapture(file_path)
            frames = []
            while cap.isOpened() and nf < 16:
                nf += 1
                ret, frame = cap.read()
                if not ret:
                    break
                frame = frame.astype(np.float32)
                frame /= 255
                # resize 到 224*224
                resized_frame = cv2.resize(frame, output_size)
                frames.append(resized_frame)
            cap.release()
            #print(file_name,nf)
            
            # 将所有帧组合成一个张量
            video_tensor = np.stack(frames)
            video_tensors.append(video_tensor)
            #print(video_tensor.shape)

    # 将所有张量连接到一起
    final_tensor = np.concatenate(video_tensors, axis=0)
    #print(final_tensor.shape)
    final_tensor = final_tensor.reshape(-1,16,224,224,3)
    
    print("Final tensor shape:", final_tensor.shape)

    return final_tensor
    


def fvd_simple_old(dir_videos,dir_ref_vid,dir_results):
    
    log_file_path = dir_results
    filename = f"/fvd_simple_record.txt"
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
    
    #num_videos = 101
    num_videos = 100    #100
    video_len = 16
    fvd_time = 20          #20
    batch_size = 20
    
    logging.info(f"num_videos:{num_videos}")

    #videos_fake = np.random.RandomState(seed_fake).rand(num_videos, video_len, 224, 224, 3).astype(np.float32)
    #videos_real = np.random.RandomState(seed_real).rand(num_videos, video_len, 224, 224, 3).astype(np.float32)
    videos_fake = read_generated(base_path=dir_videos)

    fvd_result = []
    for i in range(fvd_time):
        print(f'Computing FVD {i+1}/{fvd_time} ...')
        videos_real = read_ucf101_simple(base_path=dir_ref_vid)
        our_fvd_result = 0
        #vshape = [-1,16,224,224,3]
        our_fvd_result = compute_our_fvd(videos_fake, 
                                        videos_real, 
                                        batch_size,
                                        num_videos,
                                        device='cuda')
        print(f'[FVD scores]. Ours: {our_fvd_result}')
        logging.info(f'[FVD scores]. Ours: {our_fvd_result}')
        fvd_result.append(our_fvd_result)
    
    fvd_mean = np.mean(fvd_result)
    fvd_std = np.std(fvd_result)
    
    print(f'[Final FVD scores]. mean:{fvd_mean}, std:{fvd_std}')
    logging.info(f'[Final FVD scores]. mean:{fvd_mean}, std:{fvd_std}')
    
    
def fvd_full(dir_videos,dir_ref_vid,dir_results):
    
    log_file_path = dir_results
    filename = f"/fvd_simple_record.txt"
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
    
    #num_videos = 101
    num_videos = 2048    #100
    video_len = 16
    fvd_time = 1          #20
    batch_size = 50
    
    logging.info(f"num_videos:{num_videos}")

    #videos_fake = np.random.RandomState(seed_fake).rand(num_videos, video_len, 224, 224, 3).astype(np.float32)
    #videos_real = np.random.RandomState(seed_real).rand(num_videos, video_len, 224, 224, 3).astype(np.float32)
    videos_fake = read_generated(base_path=dir_videos)

    fvd_result = []
    for i in range(fvd_time):
        print(f'Computing FVD {i+1}/{fvd_time} ...')
        videos_real = read_ucf101_full(base_path=dir_ref_vid)
        our_fvd_result = 0
        #vshape = [-1,16,224,224,3]
        our_fvd_result = compute_our_fvd(videos_fake, 
                                        videos_real, 
                                        batch_size,
                                        num_videos,
                                        device='cuda')
        print(f'[FVD scores]. Ours: {our_fvd_result}')
        logging.info(f'[FVD scores]. Ours: {our_fvd_result}')
        fvd_result.append(our_fvd_result)
    
    fvd_mean = np.mean(fvd_result)
    fvd_std = np.std(fvd_result)
    
    print(f'[Final FVD scores]. mean:{fvd_mean}, std:{fvd_std}')
    logging.info(f'[Final FVD scores]. mean:{fvd_mean}, std:{fvd_std}')


def fvd_simple(dir_videos,dir_ref_vid,dir_results):
    
    log_file_path = dir_results
    filename = f"/fvd_simple_record.txt"
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
    
    #num_videos = 101
    num_videos = 100    #100
    video_len = 16
    fvd_time = 10          #20
    batch_size = 20
    
    logging.info(f"num_videos:{num_videos}")

    #videos_fake = np.random.RandomState(seed_fake).rand(num_videos, video_len, 224, 224, 3).astype(np.float32)
    #videos_real = np.random.RandomState(seed_real).rand(num_videos, video_len, 224, 224, 3).astype(np.float32)
    videos_fake = read_generated(base_path=dir_videos)

    fvd_result = []
    for i in range(fvd_time):
        videos_real = read_generated(base_path=os.path.join(dir_ref_vid, str(i)))
        print(f'Computing FVD {i+1}/{fvd_time} ...')
        our_fvd_result = 0
        #vshape = [-1,16,224,224,3]
        our_fvd_result = compute_our_fvd(videos_fake, 
                                        videos_real, 
                                        batch_size,
                                        num_videos=videos_fake.shape[0],
                                        device='cuda'
                                        )
        print(f'[FVD scores]. Ours: {our_fvd_result}')
        logging.info(f'[FVD scores]. Ours: {our_fvd_result}')
        fvd_result.append(our_fvd_result)
    
    fvd_mean = np.mean(fvd_result)
    fvd_std = np.std(fvd_result)
    
    print(f'[Final FVD scores]. mean:{fvd_mean}, std:{fvd_std}')
    logging.info(f'[Final FVD scores]. mean:{fvd_mean}, std:{fvd_std}')
    
def fp_fvd(dir_videos,dir_ref_vid,dir_results):
    
    log_file_path = dir_results
    filename = f"/fvd_simple_record.txt"
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
    
    num_videos = 10
    #num_videos = 100    #100
    video_len = 16
    fvd_time = 1          #20
    #batch_size = 10
    batch_size = 1

    
    logging.info(f"num_videos:{num_videos}")

    #videos_fake = np.random.RandomState(seed_fake).rand(num_videos, video_len, 224, 224, 3).astype(np.float32)
    #videos_real = np.random.RandomState(seed_real).rand(num_videos, video_len, 224, 224, 3).astype(np.float32)
    videos_fake = read_generated(base_path=dir_videos)

    fvd_result = []
    for i in range(fvd_time):
        videos_real = read_generated(base_path=dir_ref_vid)
        print(f'Computing FVD {i+1}/{fvd_time} ...')
        our_fvd_result = 0
        #vshape = [-1,16,224,224,3]
        our_fvd_result = compute_our_fvd(videos_fake, 
                                        videos_real, 
                                        batch_size,
                                        num_videos=videos_fake.shape[0],
                                        device='cuda'
                                        )
        print(f'[FVD scores]. Ours: {our_fvd_result}')
        logging.info(f'[FVD scores]. Ours: {our_fvd_result}')
        fvd_result.append(our_fvd_result)
    
    fvd_mean = np.mean(fvd_result)
    fvd_std = np.std(fvd_result)
    
    print(f'Final FPFVD scores:{fvd_mean}')
    logging.info(f'[Final FPFVD scores]. mean:{fvd_mean}, std:{fvd_std}')
    
    
def fvd_simp_ucf(dir_videos,dir_ref_vid,dir_results):
    
    log_file_path = dir_results
    filename = f"/fvd_simple_record.txt"
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
    
    num_videos = 6
    #num_videos = 100    #100
    video_len = 16
    fvd_time = 1          #20
    #batch_size = 10
    batch_size = 1
    
    logging.info(f"num_videos:{num_videos}")

    #videos_fake = np.random.RandomState(seed_fake).rand(num_videos, video_len, 224, 224, 3).astype(np.float32)
    #videos_real = np.random.RandomState(seed_real).rand(num_videos, video_len, 224, 224, 3).astype(np.float32)
    videos_fake = read_generated(base_path=dir_videos)

    fvd_result = []
    for i in range(fvd_time):
        videos_real = read_generated(base_path=dir_ref_vid)
        print(f'Computing FVD {i+1}/{fvd_time} ...')
        our_fvd_result = 0
        #vshape = [-1,16,224,224,3]
        our_fvd_result = compute_our_fvd(videos_fake, 
                                        videos_real, 
                                        batch_size,
                                        num_videos=videos_fake.shape[0],
                                        device='cuda'
                                        )
        print(f'[FVD scores]. Ours: {our_fvd_result}')
        logging.info(f'[FVD scores]. Ours: {our_fvd_result}')
        fvd_result.append(our_fvd_result)
    
    fvd_mean = np.mean(fvd_result)
    fvd_std = np.std(fvd_result)
    
    print(f'Final FVD-UCF scores:{fvd_mean}')
    logging.info(f'[Final FVD-UCF scores]. mean:{fvd_mean}, std:{fvd_std}')


if __name__ == "__main__":
    print("doing fvd")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_videos", type=str, default='', help="Specify the path of generated videos")
    parser.add_argument("--dir_ref_vid", type=str, default='', help="Specify the path of original videos")
    parser.add_argument("--dir_results", type=str, default='', help="Specify the path of results")
    parser.add_argument("--mode", type=str, default='', help="Select computing mode")
    args = parser.parse_args()
    
    if args.mode == "simple":
        fvd_simple(args.dir_videos,
                args.dir_ref_vid,
                args.dir_results)
    elif args.mode == "full":
        fvd_full(args.dir_videos,
                args.dir_ref_vid,
                args.dir_results)
    elif args.mode == "fpfvd":
        fp_fvd(args.dir_videos,
                args.dir_ref_vid,
                args.dir_results)
    elif args.mode == "simp_ucf":
        fvd_simp_ucf(args.dir_videos,
                args.dir_ref_vid,
                args.dir_results)
    else:
        print("Error: invalid FVD computing mode")
