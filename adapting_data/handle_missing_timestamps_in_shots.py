import pickle
import numpy as np
import sys
import glob
import tqdm
import os.path
from pathlib import Path


detections_dir = "/home/acances/Data/Ava_v2.2/corrected_detectron2_detections"
shots_dir = "/home/acances/Data/Ava_v2.2/corrected_shots"
new_shots_dir = "/home/acances/Data/Ava_v2.2/corrected_new_shots"


def detections_exist_for(cat, video_id, timestamp):
    detections_file = "{}/{}/{}/{:05d}_dets.pkl".format(detections_dir, cat, video_id, timestamp)
    return os.path.isfile(detections_file)


def handle_missing_timestamps(shots_file):
    video_id = shots_file.split("/")[-1].split(".")[0][6:]
    cat = shots_file.split("/")[-2]
    assert os.path.isfile(shots_file)

    boundaries = []
    with open(shots_file, "r") as f:
        for line in f:
            start, end = line.strip().split(",")
            t1, n1 = tuple(map(int, start.split("_")))
            t2, n2 = tuple(map(int, end.split("_")))
            boundaries.append([(t1, n1), (t2, n2)])
    
    max_n_index = max([max(start[1], end[1]) for start, end in boundaries])
    
    new_boundaries = []
    for start, end in boundaries:
        t1, n1 = start
        t2, n2 = end
        
        t = t1
        while True:
            while not(detections_exist_for(cat, video_id, t)) and t <= t2:
                t += 1
            if t > t2: break
            # Here, t <= t2 is the first (new) timestamp for which detections exist
            begin = t
            n_begin = n1 if begin == t1 else 1
            if n_begin == 0:
                n_begin = 1

            while detections_exist_for(cat, video_id, t) and t <= t2:
                t += 1
            
            # Here, t is the first index after begin for which no detections exist or which is > t2
            end = t - 1
            n_end = n2 if end == t2 else max_n_index
            if n_end == 0:
                n_end = 1

            new_boundaries.append([(begin, n_begin), (end, n_end)])

    new_shots_file = new_shots_dir + shots_file[len(shots_dir):]
    Path("/".join(new_shots_file.split("/")[:-1])).mkdir(parents=True, exist_ok=True)
    with open(new_shots_file, "w") as f:
        for start, end in new_boundaries:
            t1, n1 = start
            t2, n2 = end
            f.write("{:05d}_{:06d},{:05d}_{:06d}\n".format(t1, n1, t2, n2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Confirm by providing 'yes' as argument")
        sys.exit(1)
    confirm = sys.argv[1]
    if confirm != "yes":
        print("Confirm by providing 'yes' as argument")
        sys.exit(1)
    
    shots_files = glob.glob("{}/train/*".format(shots_dir))
    shots_files += glob.glob("{}/val/*".format(shots_dir))

    for shots_file in tqdm.tqdm(shots_files):
        handle_missing_timestamps(shots_file)
