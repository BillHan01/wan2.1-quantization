import os
import shutil

# Define folder paths
folder_a = '/mnt/public/fangtongcheng/iclr_code/ViDiT-Q/logs/exp_iclr/qdit_w6a6_test/generated_videos_opensora'  # Path to Folder A
folder_b = '/mnt/public/fangtongcheng/iclr_code/ViDiT-Q/logs/exp_iclr/qdit_w6a6_test/processed_videos'  # Path to Folder B
folder_txt = '/share/public/video_quant/wanrui/VBench/final_prompt_simple'

# Define subfolder paths
subfolders = ['subject_consistency', 'overall_consistency', 'scene']
subfolder_paths = [os.path.join(folder_b, subfolder) for subfolder in subfolders]

# Ensure the target folder and its subfolders exist; create them if they don't
if not os.path.exists(folder_b):
    os.makedirs(folder_b)

for subfolder_path in subfolder_paths:
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

# Get all files in Folder A, sorted by name
files = sorted([f for f in os.listdir(folder_a) if f.startswith('sample_') and f.endswith('.mp4')],
               key=lambda x: int(x.split('_')[1].split('.')[0]))

# Define partition points
cut1 = 72
cut2 = 72 + 93

# Move files to corresponding subfolders
for i, file in enumerate(files):
    src_path = os.path.join(folder_a, file)
    if i < cut1:
        dest_path = os.path.join(subfolder_paths[0], file)
    elif i < cut2:
        dest_path = os.path.join(subfolder_paths[1], file)
    else:
        dest_path = os.path.join(subfolder_paths[2], file)
    shutil.copy(src_path, dest_path)

# Rename files in each subfolder
for subfolder, subfolder_path in zip(subfolders, subfolder_paths):
    txt_file_path = os.path.join(folder_txt, f"{subfolder}.txt")

    # Read new names from the TXT file
    with open(txt_file_path, 'r') as txt_file:
        new_names = txt_file.read().splitlines()

    # Get all files in the subfolder, sorted by name
    subfolder_files = sorted([f for f in os.listdir(subfolder_path) if f.endswith('.mp4')],
                             key=lambda x: int(x.split('_')[1].split('.')[0]))

    # Check if the number of new names matches the number of files
    if len(new_names) != len(subfolder_files):
        print(f"Error: The number of files in {subfolder_path} does not match the number of lines in {txt_file_path}.")
        continue

    # Rename files
    for old_name, new_name in zip(subfolder_files, new_names):
        old_file_path = os.path.join(subfolder_path, old_name)
        new_file_path = os.path.join(subfolder_path, f"{new_name}.mp4")
        os.rename(old_file_path, new_file_path)

print("Files have been successfully categorized, moved, and renamed.")