import pickle
import numpy as np
import sys
import glob
import tqdm
from pathlib import Path

from check_detections_redundancy import check_redundancy

root = "/home/acances/Data/Ava_v2.2/detectron2_detections"
output_dir = "/home/acances/Code/ava/redundancies"
Path(output_dir).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    cat = sys.argv[1]
    detections_dir = "{}/{}".format(root, cat)
    if cat not in ["train", "val"]:
        print("Category must be train or val")
    video_ids = [e.split("/")[-1] for e in glob.glob(detections_dir + "/*")]
    for video_id in tqdm.tqdm(video_ids):
        check_redundancy(video_id, output_dir, cat)
