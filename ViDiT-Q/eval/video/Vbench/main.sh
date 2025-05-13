export HF_ENDPOINT=https://hf-mirror.com
#cd $YOUR_PATH/diffuser-dev/clean_eval/Vbench # Location of evaluate.py file
#export VBENCH_CACHE_DIR="/share/public/video_quant/vm_ckpt/" # Local download directory

current_time=$(date +"%Y%m%d%H%M%S")
JSON="/share/public/video_quant/wanrui/VBench/evaluation_results/human_action_full_info.json"
#VIDEO_PATH="/share/public/video_quant/wanrui/VBench/our_video/mix_66"
#OUTPUT="./evaluation_results/mix_66_${current_time}"
VIDEO_PATH="/share/public/video_quant/wanrui/VBench/our_video/qdiff_88" # Video directory
OUTPUT="./evaluation_results/qdiff_88_${current_time}" # Output results directory

# If torch.hub.load reports an error, try using local download by setting --load_ckpt_from_local to True
python evaluate.py --output_path $OUTPUT --full_json_dir $JSON --dimension "aesthetic_quality" --videos_path "$VIDEO_PATH/overall_consistency"
python evaluate.py --output_path $OUTPUT --full_json_dir $JSON --dimension 'scene' --videos_path "$VIDEO_PATH/scene"
python evaluate.py --output_path $OUTPUT --full_json_dir $JSON --dimension "imaging_quality" --videos_path "$VIDEO_PATH/overall_consistency"
python evaluate.py --output_path $OUTPUT --full_json_dir $JSON --dimension "overall_consistency" --videos_path "$VIDEO_PATH/overall_consistency"
#python evaluate.py --output_path $OUTPUT --full_json_dir $JSON --dimension 'human_action' --videos_path "$VIDEO_PATH/human_action"
python evaluate.py --output_path $OUTPUT --full_json_dir $JSON --dimension 'background_consistency' --videos_path "$VIDEO_PATH/scene"
#python evaluate.py --output_path $OUTPUT --full_json_dir $JSON --dimension 'temporal_flickering' --videos_path "$VIDEO_PATH/temporal_flickering"
#python evaluate.py --output_path $OUTPUT --full_json_dir $JSON --dimension 'multiple_objects' --videos_path "$VIDEO_PATH/multiple_objects"
python evaluate.py --output_path $OUTPUT --full_json_dir $JSON --dimension 'subject_consistency' --videos_path "$VIDEO_PATH/subject_consistency" --load_ckpt_from_local True
#python evaluate.py --output_path $OUTPUT --full_json_dir $JSON --dimension 'subject_consistency' --videos_path "$VIDEO_PATH/subject_consistency"
python evaluate.py --output_path $OUTPUT --full_json_dir $JSON --dimension 'dynamic_degree' --videos_path "$VIDEO_PATH/subject_consistency"
python evaluate.py --output_path $OUTPUT --full_json_dir $JSON --dimension 'motion_smoothness' --videos_path "$VIDEO_PATH/subject_consistency"
#python evaluate.py --output_path $OUTPUT --full_json_dir $JSON --dimension 'object_class' --videos_path "$VIDEO_PATH/object_class"
#python evaluate.py --output_path $OUTPUT --full_json_dir $JSON --dimension 'color' --videos_path "$VIDEO_PATH/color"
#python evaluate.py --output_path $OUTPUT --full_json_dir $JSON --dimension 'appearance_style' --videos_path "$VIDEO_PATH/appearance_style"
#python evaluate.py --output_path $OUTPUT --full_json_dir $JSON --dimension 'temporal_style' --videos_path "$VIDEO_PATH/temporal_style"
#python evaluate.py --output_path $OUTPUT --full_json_dir $JSON --dimension 'spatial_relationship' --videos_path "$VIDEO_PATH/spatial_relationship"