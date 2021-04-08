import pickle
import numpy as np
import sys
import glob
import tqdm
from pathlib import Path


detections_dir = "/home/acances/Data/Ava_v2.2/corrected_detectron2_detections"


def verify_lengths(video_id, cat):
    detections_files = sorted(glob.glob("{}/{}/{}/*".format(detections_dir, cat, video_id)))

    # List the values taken by the lengths of the different detections files
    nb_entries_values = []
    for detections_file in detections_files:
        with open(detections_file, "rb") as f: data = pickle.load(f)
        nb_entries = len(data)
        assert list(data.keys()) == list(range(1, len(data) + 1))
        if nb_entries not in nb_entries_values:
            nb_entries_values.append(nb_entries)
    
    if len(nb_entries_values) > 1:
        print(video_id)


if __name__ == "__main__":
    cat = sys.argv[1]
    video_ids = [e.split("/")[-1] for e in glob.glob("{}/{}/*".format(detections_dir, cat))]
    for video_id in tqdm.tqdm(video_ids):
        verify_lengths(video_id, cat)
