import pickle
import numpy as np
import sys
import glob
import tqdm


redundancies_dir = "/home/acances/Code/ava/redundancies"
detections_dir = "/home/acances/Data/Ava_v2.2/corrected_detectron2_detections"


def correct_detections(redundancy_file):
    video_id = redundancy_file.split("/")[-1][:11]
    cat = redundancy_file.split("/")[-2]
    redundancies = []
    with open(redundancy_file, "r") as f:
        for line in f:
            timestamp, index = line.strip().split(",")
            index = int(index)
            redundancies.append([timestamp, index])
    
    for timestamp, index in redundancies:
        detections_file = "{}/{}/{}/{}_dets.pkl".format(detections_dir, cat, video_id, timestamp)

        with open(detections_file, "rb") as f:
            data = pickle.load(f)
        
        N = len(data)
        new_data = {}
        for i in range(1, index):
            new_data[i] = data[i]
        for i in range(index, N):
            new_data[i] = data[i + 1]
        assert len(new_data) == N - 1

        with open(detections_file, "wb") as f:
            pickle.dump(new_data, f)


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
        correct_detections(redundancy_file)
