import pickle
import numpy as np
import sys
import glob
import tqdm


root = "/home/acances/Data/Ava_v2.2/detectron2_detections/train/"

video_ids = [e.split("/")[-1] for e in glob.glob(root + "/*")]

for video_id in tqdm.tqdm(video_ids):
    detection_files = glob.glob(root + video_id + "/*")
    for file in detection_files:
        with open(file, "rb") as f:
            data = pickle.load(f)

