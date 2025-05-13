dir_videos=/mnt/public/xierui/pythonprogram/HunyuanVideo_original/tmp/HunyuanVideo_DLFRGen/results/9/t0.7_k15


CUDA_VISIBLE_DEVICES=1 python3 clip_score.py --dir_videos $dir_videos --dir_prompts $dir_prompts --dir_results $dir_results --metric 'clip_temp_score'