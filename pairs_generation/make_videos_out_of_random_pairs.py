import os
import sys
import numpy as np
import cv2
import glob
import tqdm
import pickle
import random
from pathlib import Path


pos_pairs_dir = "/home/acances/Data/Ava_v2.2/random_AVA_pairs/positive"
medneg_pairs_dir = "/home/acances/Data/Ava_v2.2/random_AVA_pairs/medium_negative"

pos_videos_dir = "/home/acances/Data/Ava_v2.2/random_AVA_pairs/videos/positive"
Path(pos_videos_dir).mkdir(parents=True, exist_ok=True)
medneg_videos_dir = "/home/acances/Data/Ava_v2.2/random_AVA_pairs/videos/medium_negative"
Path(medneg_videos_dir).mkdir(parents=True, exist_ok=True)


def make_video(name, subdir, output_dir):
    subdir_name = subdir.split("/")[-1]
    video_file = "{}/{}_{}.avi".format(output_dir, subdir_name, name)

    images = sorted([img for img in os.listdir(subdir) if name in img], key=lambda img: int(img.split(".")[0].split("_")[-1]))
    frame = cv2.imread(os.path.join(subdir, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_file, 0, 8, (width,height))
    for image in images:
        video.write(cv2.imread(os.path.join(subdir, image)))
    cv2.destroyAllWindows()
    video.release()



def make_pos_videos():
    subdirs = glob.glob(pos_pairs_dir + "/*")
    for subdir in tqdm.tqdm(subdirs, leave=True):
        subdir_name = subdir.split("/")[-1]
        
        make_video("tensor1", subdir, pos_videos_dir)
        make_video("tensor2", subdir, pos_videos_dir)

def make_medneg_videos():
    subdirs = glob.glob(medneg_pairs_dir + "/*")
    for subdir in tqdm.tqdm(subdirs, leave=True):
        subdir_name = subdir.split("/")[-1]
        
        make_video("tensor1", subdir, medneg_videos_dir)
        make_video("tensor2", subdir, medneg_videos_dir)


if __name__ == "__main__":
    make_pos_videos()
    make_medneg_videos()
