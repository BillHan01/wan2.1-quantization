import os
import sys
import re


def read_log_file(file_path):
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
        return lines
    except FileNotFoundError:
        print("File not found.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <log_file_path>")
    else:
        log_file_path = os.path.realpath(sys.argv[1])
        print(log_file_path)
        lines = read_log_file(log_file_path)

        pattern_clip_temp = re.compile(r"Final average clip_temp_score: (\d+\.\d+)")
        pattern_clip = re.compile(r"Final average clip_score: (\d+\.\d+)")
        pattern_flow = re.compile(r"Final average flow_score: (\d+\.\d+)")
        # pattern_inception = re.compile(r"Inception score of generated_videos_opensora: (\d+\.\d+), std: (\d+\.\d+)")
        pattern_vqa_a = re.compile(r"Aesthetic: (\d+\.\d+)")
        pattern_vqa_t = re.compile(r"Technical: (\d+\.\d+)")
        # pattern_fvd_ucf = re.compile(r"Final FVD-UCF scores:(\d+\.\d+)")
        pattern_fvd_fp16 = re.compile(r"Final FPFVD scores:(\d+\.\d+)")
        pattern_flicker = re.compile(r"temporal flickering score:(\d+\.\d+)")
        patterns = {
            "clip_temp": pattern_clip_temp,
            "clip": pattern_clip,
            "flow": pattern_flow,
            #'inception': pattern_inception,
            "vqa_a": pattern_vqa_a,
            'vqa_t': pattern_vqa_t,
            # "fvd_ucf": pattern_fvd_ucf,
            "fvd_fp16": pattern_fvd_fp16,
            "flicker": pattern_flicker,
        }
        metrics_d = {}

        for line in lines:
            # print(line)
            for k, v in patterns.items():
                match = v.search(line)
                if match:
                    # print(match[0])
                    score = float(match.group(1))
                    metrics_d[k] = score
                    print("Matched {}: {:.3f}".format(k, score))
        print(metrics_d)
