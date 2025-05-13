import numpy as np
import cv2
import os
import argparse

def get_frames(video_path):
        frames = []
        video = cv2.VideoCapture(video_path)
        print(video_path)
        while video.isOpened():
            success, frame = video.read()
            if success:
                frames.append(frame)
            else:
                break
        video.release()
        assert frames != []
        return frames


def mae_seq(frames):
    ssds = []
    for i in range(len(frames)-1):
        ssds.append(calculate_mae(frames[i], frames[i+1]))
    return np.array(ssds)


def calculate_mae(img1, img2):
    """Computing the mean absolute error (MAE) between two images."""
    if img1.shape != img2.shape:
        print("Images don't have the same shape.")
        return
    return np.mean(cv2.absdiff(np.array(img1, dtype=np.float32), np.array(img2, dtype=np.float32)))


def cal_score(video_path):
    """please ensure the video is static"""
    frames = get_frames(video_path)
    score_seq = mae_seq(frames)
    return (255.0 - np.mean(score_seq).item())/255.0


def temporal_flickering(video_list):
    sim = []
    video_results = []
    for video_path in video_list:
        try:
            score_per_video = cal_score(video_path)
        except AssertionError:
            continue
        video_results.append({'video_path': video_path, 'video_results': score_per_video})
        sim.append(score_per_video)
    avg_score = np.mean(sim)
    return avg_score, video_results


def compute_temporal_flickering(dir, device="cuda:0"):
    files_and_dirs = os.listdir(dir)
    files = sorted([f for f in files_and_dirs if os.path.isfile(os.path.join(dir, f)) and f.endswith('.mp4')])
    video_list = [os.path.abspath(os.path.join(dir, f)) for f in files]
    all_results, video_results = temporal_flickering(video_list)
    print(f"temporal flickering score:{all_results}")
    return all_results, video_results

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_videos", type=str, default='', help="Specify the path of generated videos")
    #parser.add_argument("--dir_results", type=str, default='', help="Specify the path of results")
    args = parser.parse_args()
    
    #dir = "/share/public/video_quant/wanrui/VBench/our_video/ucf_sq_d_48"
    compute_temporal_flickering(args.dir_videos)