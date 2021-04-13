import os
import sys
import numpy as np
import cv2
import glob
import tqdm
from pathlib import Path
import pickle


tracks_dir = "/home/acances/Data/Ava_v2.2/tracks"


#  PARAMETERS
TEMP_INTERSECTION_THRESHOLD = 16
IOU_THRESHOLD = 0.2


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


def max_iou2d(tube1, tube2):
    """Compute the maximum frame IoU between two tubes with same temporal extent"""
    # tube1 : [[x1, y1, x2, y2], ...]
    
    assert tube1.shape[0] == tube2.shape[0]

    overlap = overlap2d(tube1, tube2)
    iou = overlap / (area2d(tube1) + area2d(tube2) - overlap)
    max_iou = np.max(iou)

    return max_iou


def is_medium_negative_pair(tr1, tr2):
    """Returns whether tracks tr1 and tr2 are a medium negative"""
    # tr1 : [[frame_idx, x1, y1, x2, y2, score], ...]

    # First check length of temporal intersection
    b1, e1 = tuple(map(int, tr1[[0, -1], 0]))
    b2, e2 = tuple(map(int, [tr2[0][0], tr2[-1][0]]))

    b = max(b1, b2)
    e = min(e1, e2)

    temporal_intersection = max(0, e - b + 1)
    if temporal_intersection < TEMP_INTERSECTION_THRESHOLD:
        return False
    
    # Then check maximum IOU
    tube1 = tr1[b - b1:(e + 1) - b1, 1:5]
    tube2 = tr2[b - b2:(e + 1) - b2, 1:5]

    max_iou = max_iou2d(tube1, tube2)
    return max_iou <= IOU_THRESHOLD


def find_medium_negative_pairs_for_shot(file):
    with open(file, "rb") as f:
        tracks = pickle.load(f)
    nb_tracks = len(tracks)

    nb_pairs = int(nb_tracks * (nb_tracks - 1) / 2)
    nb_medium_negative_pairs = 0
    for i in range(nb_tracks):
        tr1, sc1 = tracks[i]
        for j in range(i + 1, nb_tracks):
            tr2, sc2 = tracks[j]
            if is_medium_negative_pair(tr1, tr2):
                nb_medium_negative_pairs += 1
    # print("{}/{}".format(nb_medium_negative_pairs, nb_pairs))


def find_medium_negative_pairs_for_video(video_id, cat):
    tracks_files = glob.glob("{}/{}/{}/*".format(tracks_dir, cat, video_id))
    for tracks_file in tracks_files:
        find_medium_negative_pairs_for_shot(tracks_file)


if __name__ == "__main__":
    video_folders = glob.glob("{}/train/*".format(tracks_dir))
    video_folders += glob.glob("{}/val/*".format(tracks_dir))

    for video_folder in tqdm.tqdm(video_folders):
        video_id = video_folder.split("/")[-1]
        cat = video_folder.split("/")[-2]
        find_medium_negative_pairs_for_video(video_id, cat)
