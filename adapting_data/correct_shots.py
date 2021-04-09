import pickle
import numpy as np
import sys
import glob
import tqdm
import os.path


redundancies_dir = "/home/acances/Code/ava/redundancies"
shots_dir = "/home/acances/Data/Ava_v2.2/corrected_shots"


def correct_shots(redundancy_file):
    video_id = redundancy_file.split("/")[-1][:11]
    cat = redundancy_file.split("/")[-2]
    redundancies = {}
    nb_redundancies = 0
    with open(redundancy_file, "r") as f:
        for line in f:
            timestamp, index = tuple(map(int, line.strip().split(",")))
            redundancies[timestamp] = index
            nb_redundancies += 1
    assert len(redundancies) == nb_redundancies

    shots_file = "{}/{}/shots_{}.csv".format(shots_dir, cat, video_id)
    assert os.path.isfile(shots_file)

    boundaries = []
    with open(shots_file, "r") as f:
        for line in f:
            start, end = line.strip().split(",")
            t1, n1 = tuple(map(int, start.split("_")))
            t2, n2 = tuple(map(int, end.split("_")))
            if t1 in redundancies:
                index = redundancies[t1]
                if n1 > index:
                    n1 -= 1
            if t2 in redundancies:
                index = redundancies[t2]
                if n2 > index:
                    n2 -= 1
            boundaries.append([(t1, n1), (t2, n2)])

    with open(shots_file, "w") as f:
        for start, end in boundaries:
            t1, n1 = start
            t2, n2 = end
            f.write("{:05d}_{:06d},{:05d}_{:06d}\n".format(t1, n1, t2, n2))


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
        correct_shots(redundancy_file)
