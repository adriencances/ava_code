import os
import sys
import numpy as np
import cv2
import glob
import tqdm
from pathlib import Path
import pickle
from os.path import expanduser
import os.path as osp


sys.path.append("/home/acances/Code/tracker")
from people_detector import SortTracker


# for reproducibility
np.random.seed(0)

shots_dir = "/home/acances/Data/Ava_v2.2/final_shots"
detections_dir = "/home/acances/Data/Ava_v2.2/final_detectron2_detections"
tracks_dir = "/home/acances/Data/Ava_v2.2/tracks_SORT"


def get_shots(shots_file):
    shots = []
    with open(shots_file, "r") as f:
        for line in f:
            start, end = line.strip().split(",")
            t1, n1 = tuple(map(int, start.split("_")))
            t2, n2 = tuple(map(int, end.split("_")))
            shots.append([t1, n1, t2, n2])
    return shots


def get_cat_and_video_id(shots_file):
    cat = shots_file.split("/")[-2]
    video_id = shots_file.split("/")[-1].split(".")[0][6:]
    return cat, video_id


def compute_full_tracks_of_shot(shot, video_id, cat, shot_id):
    t1, n1, t2, n2 = shot
    # Gather the detections from all needed timestamps (indices t1, t1+1, ..., t2)
    all_detections = []
    N = None
    for timestamp in range(t1, t2 + 1):
        detections_pkl = "{}/{}/{}/{:05d}_dets.pkl".format(detections_dir, cat, video_id, timestamp)
        with open(detections_pkl, "rb") as f:
            detections = pickle.load(f)
        all_detections += list(detections.values())
        if N is None: N = len(detections)
        assert len(detections) == N
    
    for i in range(len(all_detections)):
        if all_detections[i].shape[0] == 0:
            all_detections[i] = np.empty((0, 5))
    
    starting_frame = n1 - 1
    ending_frame = (t2 - t1)*N + (n2 - 1)
    assert ending_frame <= len(all_detections)

    # Remove detections for frame outside the shot window
    for i in range(starting_frame):
        all_detections[i] = np.empty((0, 5))
    for i in range(ending_frame, len(all_detections)):
        all_detections[i] = np.empty((0, 5))

    bboxes = [e[:,:4] if len(e) > 0 else np.empty((0, 4)) for e in all_detections]
    scores = [e[:,-1] if len(e) > 0 else np.empty((0, 1)) for e in all_detections]

    tracker = SortTracker()
    tracks = tracker.track_bboxes(bboxes, scores)

    tracks_file = "{}/{}/{}/{:05d}_tracks.pkl".format(tracks_dir, cat, video_id, shot_id)
    Path("/".join(tracks_file.split("/")[:-1])).mkdir(parents=True, exist_ok=True)
    with open(tracks_file, "wb") as f:
        pickle.dump(tracks, f)


def compute_tracks_for_shots_file(shots_file):
    cat, video_id = get_cat_and_video_id(shots_file)
    shots = get_shots(shots_file)
    for shot_id, shot in tqdm.tqdm(list(enumerate(shots))):
        compute_full_tracks_of_shot(shot, video_id, cat, shot_id)

def compute_tracks(video_id, shot_id):
    shots_file = "{}/train/shots_{}.csv".format(shots_dir, video_id)
    shots = get_shots(shots_file)
    compute_full_tracks_of_shot(shots[shot_id], video_id, "train", shot_id)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Confirm by providing 'yes' as argument")
        sys.exit(1)
    confirm = sys.argv[1]
    if confirm != "yes":
        print("Confirm by providing 'yes' as argument")
        sys.exit(1)
    
    if os.environ['CONDA_DEFAULT_ENV'] != "tf-gpu":
        print("Use 'tf-gpu' conda environment")
        sys.exit(1)

    shots_files = glob.glob("{}/train/*".format(shots_dir))
    shots_files += glob.glob("{}/val/*".format(shots_dir))

    for shots_file in tqdm.tqdm(shots_files):
        compute_tracks_for_shots_file(shots_file)


    # video_id = sys.argv[1]
    # shot_id = int(sys.argv[2])
    # compute_tracks(video_id, shot_id)

