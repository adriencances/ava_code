import os
import sys
import numpy as np
import cv2
import glob
import tqdm
from pathlib import Path
import pickle


tracks_dir = "/home/acances/Data/Ava_v2.2/tracks"
pairs_dir = "/home/acances/Data/Ava_v2.2/pairs/"

Path(pairs_dir).mkdir(parents=True, exist_ok=True)


#  PARAMETERS
TEMP_INTERSECTION_THRESHOLD = 64
IOU_THRESHOLD = 0.2
FRAME_PROPORTION = 0.1
SEGMENT_LENGTH = 64


def area2d(b):
    """Compute the areas for a set of 2D boxes"""

    return (b[:,2]-b[:,0]) * (b[:,3]-b[:,1])

def overlap2d(b1, b2):
    """Compute the overlaps between a set of boxes b1 and one box b2"""
    # b1 : [[x1, y1, x2, y2], ...]

    assert b1.shape == b2.shape

    xmin = np.maximum(b1[:,0], b2[:,0])
    ymin = np.maximum(b1[:,1], b2[:,1])
    xmax = np.minimum(b1[:,2], b2[:,2])
    ymax = np.minimum(b1[:,3], b2[:,3])

    width = np.maximum(0, xmax - xmin)
    height = np.maximum(0, ymax - ymin)

    return width * height


def iou2d(tube1, tube2):
    """Compute the frame IoU vector of two tubes with the same temporal extent"""
    # tube1 : [[x1, y1, x2, y2], ...]
    
    assert tube1.shape[0] == tube2.shape[0]

    overlap = overlap2d(tube1, tube2)
    iou = overlap / (area2d(tube1) + area2d(tube2) - overlap)

    return iou


def get_pos_pairs_begin_indices(iou):
    """Returns the list of beginning indices of the positive segments,
    given the list of IoUs of the intersection of two tracks.
    Thus, the returned indices are relative to intersection of the two tracks."""
    is_above = iou > IOU_THRESHOLD
    N = len(is_above)
    begin_indices = []
    # Segments are chosen to be pairwise disjoint
    j = 0
    while True:
        if j + SEGMENT_LENGTH > N: break
        # Check whether or not there are enough frames with IoU above the threshold
        if np.sum(is_above[j:j+SEGMENT_LENGTH]) / SEGMENT_LENGTH >= FRAME_PROPORTION:
            begin_indices.append(j)
            j += SEGMENT_LENGTH
        else:
            j += 1
    return begin_indices


def get_medneg_pairs_begin_indices(iou):
    """Returns the list of beginning indices of the medium_negative segments,
    given the list of IoUs of the intersection of two tracks.
    Thus, the returned indices are relative to intersection of the two tracks."""
    is_above = iou > IOU_THRESHOLD
    N = len(is_above)
    begin_indices = []
    # Segments are chosen to be pairwise disjoint
    j = 0
    while True:
        if j + SEGMENT_LENGTH > N: break
        # Check whether or not all frames have IoU below the threshold
        if np.sum(is_above[j:j+SEGMENT_LENGTH]) == 0:
            begin_indices.append(j)
            j += SEGMENT_LENGTH
        else:
            j += 1
    return begin_indices


def get_pos_and_medneg_pairs_begin_frames_for_tracks(tr1, tr2):
    """Returns the list of begin_frame indices of positive pairs and of medium negative pairs.
    The indices are relative to the shot to which the two tracks belong."""
    # tr1 : [[frame_idx, x1, y1, x2, y2, score], ...]

    # Compute temporal intersection
    b1, e1 = tuple(map(int, tr1[[0, -1], 0]))
    b2, e2 = tuple(map(int, [tr2[0][0], tr2[-1][0]]))

    b = max(b1, b2)
    e = min(e1, e2)

    # If temporal intersection is too small, there can be no positive pairs
    temporal_intersection = max(0, e - b + 1)
    if temporal_intersection < TEMP_INTERSECTION_THRESHOLD:
        return [], []
    
    # Compute IoU vector to identify positive pairs
    tube1 = tr1[b - b1:(e + 1) - b1, 1:5]
    tube2 = tr2[b - b2:(e + 1) - b2, 1:5]

    iou = iou2d(tube1, tube2)
    pos_begin_indices = get_pos_pairs_begin_indices(iou)
    medneg_begin_indices = get_medneg_pairs_begin_indices(iou)

    # Shift indices so they are relative to the whole shot and not just the intersection of the tracks
    pos_begin_frames = [b + j for j in pos_begin_indices]
    medneg_begin_frames = [b + j for j in medneg_begin_indices]
    return pos_begin_frames, medneg_begin_frames


def get_pos_and_medneg_pairs_for_shot(file):
    video_id = file.split("/")[-2]
    shot_id = int(file.split("/")[-1].split("_")[0])
    with open(file, "rb") as f:
        tracks = pickle.load(f)
    nb_tracks = len(tracks)

    pos_pairs = []
    medneg_pairs = []
    for i in range(nb_tracks):
        tr1, sc1 = tracks[i]
        for j in range(i + 1, nb_tracks):
            tr2, sc2 = tracks[j]
            pos_begin_frames, medneg_begin_frames = get_pos_and_medneg_pairs_begin_frames_for_tracks(tr1, tr2)
            for begin_frame in pos_begin_frames:
                pair = []
                pair += [video_id, shot_id, i, begin_frame, begin_frame + SEGMENT_LENGTH]
                pair += [video_id, shot_id, j, begin_frame, begin_frame + SEGMENT_LENGTH]
                pos_pairs.append(pair)
            for begin_frame in medneg_begin_frames:
                pair = []
                pair += [video_id, shot_id, i, begin_frame, begin_frame + SEGMENT_LENGTH]
                pair += [video_id, shot_id, j, begin_frame, begin_frame + SEGMENT_LENGTH]
                medneg_pairs.append(pair)
    return pos_pairs, medneg_pairs


def compute_pos_and_medneg_pairs_for_video(video_id, cat):
    tracks_files = glob.glob("{}/{}/{}/*".format(tracks_dir, cat, video_id))
    video_pos_pairs = []
    video_medneg_pairs = []
    for tracks_file in tracks_files:
        pos_pairs, medneg_pairs = get_pos_and_medneg_pairs_for_shot(tracks_file)
        video_pos_pairs += pos_pairs
        video_medneg_pairs += medneg_pairs

    pos_output_file = "{}/{}/positive/pairs_{}.csv".format(pairs_dir, cat, video_id)
    medneg_output_file = "{}/{}/medium_negative/pairs_{}.csv".format(pairs_dir, cat, video_id)
    Path("/".join(pos_output_file.split("/")[:-1])).mkdir(parents=True, exist_ok=True)
    Path("/".join(medneg_output_file.split("/")[:-1])).mkdir(parents=True, exist_ok=True)

    with open(pos_output_file, "w") as f:
        for pair in video_pos_pairs:
            f.write(",".join(map(str, pair)) + "\n")
    
    with open(medneg_output_file, "w") as f:
        for pair in video_medneg_pairs:
            f.write(",".join(map(str, pair)) + "\n")


if __name__ == "__main__":
    video_folders = glob.glob("{}/train/*".format(tracks_dir))
    video_folders += glob.glob("{}/val/*".format(tracks_dir))

    for video_folder in tqdm.tqdm(video_folders):
        video_id = video_folder.split("/")[-1]
        cat = video_folder.split("/")[-2]
        compute_pos_and_medneg_pairs_for_video(video_id, cat)
