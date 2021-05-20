import os
import sys
import numpy as np
import cv2
import glob
import tqdm
from pathlib import Path
import pickle
import random as rd

from count_pairs import get_nb_pairs
from settings import Settings


tracks_dir = Settings.tracks_dir
pairs_dir = Settings.pairs_dir

Path(pairs_dir).mkdir(parents=True, exist_ok=True)


#  PARAMETERS
SEGMENT_LENGTH = Settings.SEGMENT_LENGTH


def random_element_of(L):
    if L == []:
        return None
    random_index = np.random.randint(len(L))
    return L[random_index]


def get_random_segment(video_folder):
    tracks_files = glob.glob("{}/*".format(video_folder))
    while True:
        tracks_file = rd.choice(tracks_files)
        video_id = tracks_file.split("/")[-2]
        shot_id = int(tracks_file.split("/")[-1].split("_")[0])

        with open(tracks_file, "rb") as f:
            tracks = pickle.load(f)
        interesting_track_indices = [i for i in range(len(tracks)) if len(tracks[i][0]) >= SEGMENT_LENGTH]
        if interesting_track_indices == []:
            continue

        track_id = rd.choice(interesting_track_indices)
        track = tracks[track_id][0]

        b, e = tuple(map(int, track[[0, -1], 0]))

        begin_frame = np.random.randint(b, (e + 1) - (SEGMENT_LENGTH - 1))
        return [video_id, shot_id, track_id, begin_frame, begin_frame + SEGMENT_LENGTH]


def compute_easy_negative_pairs_for_category(cat):
    video_folders = glob.glob("{}/{}/*".format(tracks_dir, cat))

    nb_positives = get_nb_pairs(cat, "positive")
    nb_pairs_wanted = nb_positives // 2
    nb_videos = len(video_folders)
    nb_video_pairs = nb_videos * (nb_videos - 1) // 2
    nb_pairs_wanted_by_video_pair = int(np.ceil(nb_pairs_wanted / nb_video_pairs))

    pairs = []
    for id1 in tqdm.tqdm(range(len(video_folders))):
        video_folder_1 = video_folders[id1]
        for id2 in range(id1 + 1, len(video_folders)):
            video_folder_2 = video_folders[id2]
            for i in range(nb_pairs_wanted_by_video_pair):
                segment_1 = get_random_segment(video_folder_1)
                segment_2 = get_random_segment(video_folder_2)
                pair = segment_1 + segment_2
                pairs.append(pair)
    
    output_file = "{}/{}/easy_negative/pairs.csv".format(pairs_dir, cat)
    Path("/".join(output_file.split("/")[:-1])).mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for pair in pairs:
            f.write(",".join(map(str, pair)) + "\n")


def compute_easy_negative_pairs():
    compute_easy_negative_pairs_for_category("train")
    compute_easy_negative_pairs_for_category("val")
