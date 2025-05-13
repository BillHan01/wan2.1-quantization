# iter through some folder to test all metrics
target_dir="$YOUR_PATH/project/attn_quant/diffuser-dev/examples/cogvideo_attn/logs/"

for folder in $(find "$target_dir" -type d -maxdepth 1 -mindepth 1 -exec basename {} \;); do
    # 将文件夹名称作为 $1 参数传入 main.sh

    echo "$target_dir"
    ./simplebench_comp_101.sh  "$target_dir/$folder/generated_videos_30"

done


