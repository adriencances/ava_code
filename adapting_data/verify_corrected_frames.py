import pickle
import numpy as np
import sys
import glob
import tqdm
from pathlib import Path


frames_dir = "/media/hdd/adrien/Ava_v2.2/correct_frames"


def verify_frames(video_folder):
    timestamp_folders = glob.glob("{}/*".format(video_folder))
    ref_nb_frames = len(glob.glob("{}/*".format(timestamp_folders[0])))

    for timestamp_folder in timestamp_folders:
        nb_frames = len(glob.glob("{}/*".format(timestamp_folder)))
        assert nb_frames == ref_nb_frames


if __name__ == "__main__":
    video_folders = glob.glob("{}/train/*".format(frames_dir))
    video_folders += glob.glob("{}/val/*".format(frames_dir))
    for video_folder in tqdm.tqdm(video_folders):
        verify_frames(video_folder)
