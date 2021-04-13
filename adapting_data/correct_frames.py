import pickle
import numpy as np
import sys
import glob
import tqdm
import os
import subprocess


redundancies_dir = "/home/acances/Code/ava/redundancies"
frames_dir = "/media/hdd/adrien/Ava_v2.2/correct_frames"


def correct_frames(redundancy_file):
    video_id = redundancy_file.split("/")[-1][:11]
    cat = redundancy_file.split("/")[-2]
    redundancies = []
    with open(redundancy_file, "r") as f:
        for line in f:
            timestamp, index = line.strip().split(",")
            index = int(index)
            redundancies.append([timestamp, index])
    
    for timestamp, index in redundancies:
        frame_folder = "{}/{}/{}/{}".format(frames_dir, cat, video_id, timestamp)
        assert os.path.isdir(frame_folder)
        frame_to_remove = "{}/{}/{}/{}/{:06d}.jpg".format(frames_dir, cat, video_id, timestamp, index)
        assert os.path.isfile(frame_to_remove)

        command = ["rm", frame_to_remove]
        subprocess.call(command)

        i = index + 1
        previous_frame = "{}/{}/{}/{}/{:06d}.jpg".format(frames_dir, cat, video_id, timestamp, i - 1)
        frame = "{}/{}/{}/{}/{:06d}.jpg".format(frames_dir, cat, video_id, timestamp, i)
        while os.path.isfile(frame):
            assert not(os.path.isfile(previous_frame))
            command = ["mv", frame, previous_frame]
            subprocess.call(command)
            i += 1
            previous_frame = "{}/{}/{}/{}/{:06d}.jpg".format(frames_dir, cat, video_id, timestamp, i - 1)
            frame = "{}/{}/{}/{}/{:06d}.jpg".format(frames_dir, cat, video_id, timestamp, i)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Confirm by providing 'yes' as argument")
        sys.exit(1)
    confirm = sys.argv[1]
    if confirm != "yes":
        print("Confirm by providing 'yes' as argument")
        sys.exit(1)

    redundancies_files = glob.glob("{}/train/*".format(redundancies_dir))
    redundancies_files += glob.glob("{}/val/*".format(redundancies_dir))
    for redundancy_file in tqdm.tqdm(redundancies_files):
        correct_frames(redundancy_file)
