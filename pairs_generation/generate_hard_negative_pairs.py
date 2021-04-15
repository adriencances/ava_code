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
SHIFT = 25


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


def get_begin_indices(iou):
    """Returns the list of beginning indices of the hard negative segments,
    given the list of IoUs of the intersection of two tracks.
    Thus, the returned indices are relative to intersection of the two tracks."""
    is_above = iou > IOU_THRESHOLD
    N = len(is_above)
    begin_indices = []
    # First look for indices to shift forward
    j = 0
    while True:
        if j + SHIFT + SEGMENT_LENGTH > N: break
        # Check whether or not there are enough frames with IoU above the threshold
        if np.sum(is_above[j:j+SEGMENT_LENGTH]) / SEGMENT_LENGTH >= FRAME_PROPORTION:
            begin_indices.append(j)
            j += SEGMENT_LENGTH
        else:
            j += 1
    # Then look for indices to shift backward
    j = SHIFT
    while True:
        if j + SEGMENT_LENGTH > N: break
        # Check whether or not there are enough frames with IoU above the threshold
        if np.sum(is_above[j:j+SEGMENT_LENGTH]) / SEGMENT_LENGTH >= FRAME_PROPORTION:
            begin_indices.append(j - SHIFT)
            j += SEGMENT_LENGTH
        else:
            j += 1
    return begin_indices

def get_hard_negative_pairs_begin_frames_for_tracks(tr1, tr2):
    """Returns the list of begin_frame indices of all hard negative pairs.
    The indices are relative to the shot to which the two tracks belong."""
    # tr1 : [[frame_idx, x1, y1, x2, y2, score], ...]

    # Compute temporal intersection
    b1, e1 = tuple(map(int, tr1[[0, -1], 0]))
    b2, e2 = tuple(map(int, [tr2[0][0], tr2[-1][0]]))

    b = max(b1, b2)
    e = min(e1, e2)

    # If temporal intersection is too small, there can be no hard negative pairs
    temporal_intersection = max(0, e - b + 1)
    if temporal_intersection < TEMP_INTERSECTION_THRESHOLD + SHIFT:
        return []
    
    # Compute IoU vector to identify hard negative pairs
    tube1 = tr1[b - b1:(e + 1) - b1, 1:5]
    tube2 = tr2[b - b2:(e + 1) - b2, 1:5]

    iou = iou2d(tube1, tube2)
    begin_indices = get_begin_indices(iou)

    # Shift indices so they are relative to the whole shot and not just the intersection of the tracks
    begin_frames = [b + j for j in begin_indices]
    return begin_frames


def get_hard_negative_pairs_for_shot(file):
    video_id = file.split("/")[-2]
    shot_id = int(file.split("/")[-1].split("_")[0])
    with open(file, "rb") as f:
        tracks = pickle.load(f)
    nb_tracks = len(tracks)

    pairs = []
    for i in range(nb_tracks):
        tr1, sc1 = tracks[i]
        for j in range(i + 1, nb_tracks):
            tr2, sc2 = tracks[j]
            begin_frames = get_hard_negative_pairs_begin_frames_for_tracks(tr1, tr2)
            for begin_frame in begin_frames:
                pair = [video_id, shot_id, i, begin_frame, begin_frame + SEGMENT_LENGTH]
                pair += [video_id, shot_id, j, (begin_frame + SHIFT), (begin_frame + SHIFT) + SEGMENT_LENGTH]
                pairs.append(pair)
                pair = [video_id, shot_id, i, (begin_frame + SHIFT), (begin_frame + SHIFT) + SEGMENT_LENGTH]
                pair += [video_id, shot_id, j, begin_frame, begin_frame + SEGMENT_LENGTH]
                pairs.append(pair)
    return pairs


def compute_hard_negative_pairs_for_video(video_id, cat):
    tracks_files = glob.glob("{}/{}/{}/*".format(tracks_dir, cat, video_id))
    pairs = []
    for tracks_file in tracks_files:
        pairs += get_hard_negative_pairs_for_shot(tracks_file)

    output_file = "{}/{}/hard_negative/pairs_{}.csv".format(pairs_dir, cat, video_id)
    Path("/".join(output_file.split("/")[:-1])).mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for pair in pairs:
            f.write(",".join(map(str, pair)) + "\n")


if __name__ == "__main__":
    video_folders = glob.glob("{}/train/*".format(tracks_dir))
    video_folders += glob.glob("{}/val/*".format(tracks_dir))

    for video_folder in tqdm.tqdm(video_folders):
        video_id = video_folder.split("/")[-1]
        cat = video_folder.split("/")[-2]
        compute_hard_negative_pairs_for_video(video_id, cat)
