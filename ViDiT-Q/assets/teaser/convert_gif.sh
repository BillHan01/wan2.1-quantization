#!/bin/bash

# 遍历当前目录下的所有 .mp4 文件
for file in *.mp4; do
    # 检查文件是否存在
    if [ -f "$file" ]; then
        # 获取文件名（不带扩展名）
        filename="${file%.*}"

        # 生成调色板
        ffmpeg -y -i "$file" -vf "fps=10,scale=320:-1:flags=lanczos,palettegen" "${filename}_palette.png"

        # 使用调色板生成优化的 GIF
        ffmpeg -y -i "$file" -i "${filename}_palette.png" -filter_complex "fps=10,scale=320:-1:flags=lanczos[x];[x][1:v]paletteuse" "${filename}.gif"

        # 删除临时调色板文件
        rm "${filename}_palette.png"

        # 输出转换完成的信息
        echo "转换完成: $file -> ${filename}.gif"
    else
        echo "未找到 .mp4 文件"
    fi
done
