import os
import sys
import numpy as np
import cv2
import glob
import tqdm
import pickle
import random
from pathlib import Path

sys.path.append("/home/acances/Code/human_interaction_SyncI3d")
from dataset_aux import FrameProcessor


class RandomAvaPairs:
    def __init__(self, phase="train", nb_pairs=100, seed=0):
        self.w = 224
        self.h = 224
        self.alpha = 0.1

        self.phase = phase
        self.frames_dir = "/media/hdd/adrien/Ava_v2.2/correct_frames"
        self.shots_dir = "/home/acances/Data/Ava_v2.2/final_shots"
        self.tracks_dir = "/home/acances/Data/Ava_v2.2/tracks"
        self.pairs_dir = "/home/acances/Data/Ava_v2.2/pairs16"

        self.output_dir = "/home/acances/Data/Ava_v2.2/random_AVA_pairs"
        Path(self.output_dir).mkdir(exist_ok=True)

        self.frame_processor = FrameProcessor(self.w, self.h, self.alpha, self.phase, self.frames_dir, self.shots_dir, self.tracks_dir)

        random.seed(seed)
        self.nb_pairs = nb_pairs
        self.gather_positive_pairs()
        self.gather_negative_pairs()

    def gather_positive_pairs(self):
        print("Gathering positive pairs")
        self.positive_pairs = []
        pairs_files = glob.glob("{}/{}/positive/*".format(self.pairs_dir, self.phase))
        for file in tqdm.tqdm(pairs_files, leave=True):
            with open(file, "r") as f:
                for line in f:
                    pair = line.strip().split(",")
                    self.positive_pairs.append(pair + [1])
        self.positive_pairs = random.sample(self.positive_pairs, self.nb_pairs)
        random.shuffle(self.positive_pairs)
    
    def gather_negative_pairs(self):
        nb_medium_negatives = self.nb_pairs
        print("Gathering medium negative pairs")
        self.medium_negative_pairs = []
        pairs_files = glob.glob("{}/{}/medium_negative/*".format(self.pairs_dir, self.phase))
        for file in tqdm.tqdm(pairs_files, leave=True):
            with open(file, "r") as f:
                for line in f:
                    pair = line.strip().split(",")
                    self.medium_negative_pairs.append(pair + [0])
        self.medium_negative_pairs = random.sample(self.medium_negative_pairs, self.nb_pairs)
        random.shuffle(self.medium_negative_pairs)
    
    def get_pair_tensors(self, pair):
        video_id1, shot_id1, i1, begin1, end1, video_id2, shot_id2, i2, begin2, end2, label = pair
        shot_id1, track_id1, begin1, end1 = list(map(int, [shot_id1, i1, begin1, end1]))
        shot_id2, track_id2, begin2, end2 = list(map(int, [shot_id2, i2, begin2, end2]))
        assert end1 - begin1 == end2 - begin2

        tensor1 = self.frame_processor.processed_frames(video_id1, shot_id1, track_id1, begin1, end1)
        tensor2 = self.frame_processor.processed_frames(video_id2, shot_id2, track_id2, begin2, end2)

        return tensor1, tensor2

    def print_pair(self, pair, category, index):
        tensor1, tensor2 = self.get_pair_tensors(pair)

        output_subdir = "{}/{}/pair_{}".format(self.output_dir, category, index)
        Path(output_subdir).mkdir(parents=True, exist_ok=True)

        for i in range(tensor1.shape[1]):
            filename1 = "{}/tensor1_frame_{}.jpg".format(output_subdir, i + 1)
            frame1 = tensor1[:,i,:,:].numpy().transpose(2, 1, 0)
            cv2.imwrite(filename1, frame1)

            filename2 = "{}/tensor2_frame_{}.jpg".format(output_subdir, i + 1)
            frame2 = tensor2[:,i,:,:].numpy().transpose(2, 1, 0)
            cv2.imwrite(filename2, frame2)
    
    def print_all_pairs(self):
        for index, pair in enumerate(tqdm.tqdm(self.positive_pairs)):
            self.print_pair(pair, "positive", index)

        for index, pair in enumerate(tqdm.tqdm(self.medium_negative_pairs)):
            self.print_pair(pair, "medium_negative", index)


if __name__ == "__main__":
    random_ava_pairs = RandomAvaPairs()
    random_ava_pairs.print_all_pairs()
