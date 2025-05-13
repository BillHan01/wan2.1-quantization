# --- Speqcify the generated images for testing ---
dir_videos="$YOUR_PATH/ViDiT-Q/examples/opensora1.2/logs/w4a8_mp/"

dir_save=$(dirname $dir_videos)    

# --- Spcify the prompts used for generation (to measure text-video alignment) ---
dir_prompts="$YOUR_PATH/ViDiT-Q/examples/opensora1.2/assets/t2v_samples.txt"  

# --- Specify the FP generated / groun-truth reference videos (for FVD and FVD-FP16 computation) ---
dir_ref_vid_fp="$YOUR_PATH/ViDiT-Q/examples/opensora1.2/logs/software_simulation_mp/"		  #fvd_fp16

# --- Specify the eval project root path ---
root_path="./"

# --- Specify the eval log path ---
dir_results="./logs"
current_time=$(date +"%Y-%m-%d_%H-%M-%S")
dir_results=$dir_results/$current_time
mkdir -p "$dir_results"

# --- Specify the GPU to run ----
GPU_ID=1

# --- clean the (possibly) existing eval results ---
if [ -f "$dir_save/metrics.log" ]; then
	echo "removing existing metrics.log"
    rm $dir_save/metrics.log
fi

if [ -f "$dir_videos/eval.txt" ]; then
	rm $dir_videos/eval.txt
fi

# -----------------  Main Evaluation -----------------------
echo "Please be noted that when certain metric measurement raises error, they will not appear in final in the terminal, plz ref to the metrics.log for error information. "

# ----------------- FVD -----------------------
#echo " --------- Evaluating the FVD with UCF-101 Features... ----------"
# DEBUG: not tested and used yet, reruiqres generating 101 videos
#cd $root_path  # shift to the main directory
#CUDA_VISIBLE_DEVICES=$GPU_ID python3 fvd.py --dir_videos $dir_videos --dir_results $dir_results --dir_ref_vid $dir_ref_vid_ucf --mode "simp_ucf" 2>&1 | tee -a $dir_save/metrics.log

# echo " ---------- Evaluating the FVD with FP16 generated videos... ------------"
CUDA_VISIBLE_DEVICES=$GPU_ID python3 fvd.py --dir_videos $dir_videos --dir_results $dir_results --dir_ref_vid $dir_ref_vid_fp --mode "fpfvd" 2>&1 | tee -a $dir_save/metrics.log

# ----------------- CLIPSIM & CLIP-Temp -----------------------
echo "--------- Evaluating the CLIPSIM/CLIPTemp... ---------------"
CUDA_VISIBLE_DEVICES=$GPU_ID python3 clip_score.py --dir_videos $dir_videos --dir_prompts $dir_prompts --dir_results $dir_results --metric 'clip_temp_score' 2>&1 |  tee -a  $dir_save/metrics.log
CUDA_VISIBLE_DEVICES=$GPU_ID python3 clip_score.py --dir_videos $dir_videos --dir_prompts $dir_prompts --dir_results $dir_results --metric 'clip_score' 2>&1 |  tee -a  $dir_save/metrics.log

# ----------------- VQA -----------------------
echo "--------- Evaluating the VQA... -------------"
CUDA_VISIBLE_DEVICES=$GPU_ID python3 evaluate_a_set_of_videos.py --dir_videos $dir_videos --dir_results $dir_results 2>&1 | tee -a  $dir_save/metrics.log

# ----------------- Flow Score -----------------------
echo "----------- Evaluating the Flow Score... -----------"
cd ./RAFT
CUDA_VISIBLE_DEVICES=$GPU_ID python3 optical_flow_scores.py --dir_videos $dir_ref_vid_fp --metric 'flow_score' --dir_results ../$dir_results 2>&1 | tee -a $dir_save/metrics.log  # dir_results need to be reverted outside ./RAFT
CUDA_VISIBLE_DEVICES=$GPU_ID python3 optical_flow_scores.py --dir_videos $dir_videos --metric 'flow_score' --dir_results ../$dir_results 2>&1 | tee -a $dir_save/metrics.log  # dir_results need to be reverted outside ./RAFT
cd ../

# ----------------- Temporal Flickering -----------------------
echo "----------- Evaluating the Temporal Flickering... -----------"
CUDA_VISIBLE_DEVICES=$GPU_ID python3 temporal_flickering.py --dir_videos $dir_videos  2>&1 | tee -a $dir_save/metrics.log

# Dump cleaned output
echo "------------------------------------------"
echo "------------ Final Results: ---------------"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python read_metric_log.py  $dir_save/metrics.log
echo "------------------------------------------"

