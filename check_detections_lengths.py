import pickle
import numpy as np
import sys
import glob
import tqdm
from pathlib import Path


detections_dir = "/home/acances/Data/Ava_v2.2/corrected_detectron2_detections"


def check_lenghts(video_folder):
    lengths_list = []
    detections_files = glob.glob(video_folder + "/*")
    for file in detections_files:
        with open(file, "rb") as f: detections = pickle.load(f)
        length = len(detections)
        if length not in lengths_list:
            lengths_list.append(length)
    
    assert len(lengths_list) == 1


if __name__ == "__main__":
    video_folders = glob.glob("{}/train/*".format(detections_dir))
    video_folders += glob.glob("{}/val/*".format(detections_dir))

    for video_folder in tqdm.tqdm(video_folders):
        check_lenghts(video_folder)
