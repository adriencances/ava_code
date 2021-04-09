import pickle
import numpy as np
import sys
import glob
import tqdm
from pathlib import Path


def check_redundancy(video_id, output_dir, cat):
    detections_root = "/home/acances/Data/Ava_v2.2/detectron2_detections/{}/".format(cat)
    detections_path = detections_root + video_id + "/"
    detections_files = sorted(glob.glob(detections_path + "*"))

    output_file = "{}/{}/{}_redundancies.csv".format(output_dir, cat, video_id)
    Path("{}/{}".format(output_dir, cat)).mkdir(parents=True, exist_ok=True)

    # List the values taken by the lengths of the different detections files
    nb_entries_values = []
    for detections_file in detections_files:
        timestamp = detections_file.split("/")[-1].split("_")[0]
        with open(detections_file, "rb") as f: data = pickle.load(f)
        nb_entries = len(data)
        if nb_entries not in nb_entries_values:
            nb_entries_values.append(nb_entries)
    
    # Get the most likely number of frames per second
    # (integer which should match the number of extracted frames per timestamp)
    # and check that all detecions file lenghts are either nb_frames or nb_frames + 1
    nb_frames = min(nb_entries_values)
    if len(nb_entries_values) > 1 and sorted(nb_entries_values) != [nb_frames, nb_frames + 1]:
        print("ATTENTION: \t", timestamp, "\t nb_entries > nb_frames + 1")
    
    # List the timestamps of length != nb_frames for which there are no consecutive redundant indices
    interesting_timestamps = []
    all_redundancies = []
    for detections_file in detections_files:
        timestamp = detections_file.split("/")[-1].split("_")[0]
        with open(detections_file, "rb") as f: data = pickle.load(f)
        nb_entries = len(data)
        if nb_entries != nb_frames:
            # if nb_entries != nb_frames + 1:
            #     print("ATTENTION: \t", timestamp, "\t nb_entries > nb_frames + 1")
            nb_redundancies = 0
            non_empty_redundances = 0
            redundant_index = None
            for i in range(1, nb_entries):
                if len(data[i]) == len(data[i+1]) and np.sum(np.abs(data[i] - data[i+1])) == 0:
                    nb_redundancies += 1
                    if data[i] != []:
                        non_empty_redundances += 1
                        redundant_index = i
                        # print("VOILA \t", video_id, timestamp)
            if nb_redundancies < nb_entries - nb_frames:
                interesting_timestamps.append(timestamp)
            
            if non_empty_redundances == 0:
                redundant_index = 1
                # print("VOILA \t", video_id, timestamp)
            
            all_redundancies.append([timestamp, redundant_index])

            # if not (len(data[1]) == len(data[2]) and np.sum(np.abs(data[1] - data[2])) == 0):
            #     interesting_timestamps.append(timestamp)

    with open(output_file, "w") as f:
        for timestamp, index in all_redundancies:
            f.write(",".join([timestamp, str(index)]) + "\n")

    # if interesting_timestamps != []:
    #     print("Problem with \t", video_id, "\t Nb interesting timestamps: \t", len(interesting_timestamps))
    #     with open(output_file, "a") as f:
    #         f.write(",".join([video_id, str(nb_frames), str(len(interesting_timestamps))]) + "\n")
    #         f.write(",".join(interesting_timestamps) + "\n")

    # print("\t".join(interesting_timestamps))
    # print("Nb interesting timestamps: \t", len(interesting_timestamps))
    # print("Nb entries: \t", "\t".join(map(str, nb_entries_values)))    
