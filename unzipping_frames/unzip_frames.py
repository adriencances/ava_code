import subprocess
import sys
import glob
import tqdm
from pathlib import Path


zipped_frames_dir = "/media/hdd/datasets/AVAv2.2/vic_frames"
frames_dir = "/media/hdd/adrien/Ava_v2.2/frames"

Path(frames_dir).mkdir(parents=True, exist_ok=True)


def unzip(tar_gz_file, cat):
    target_dir = "{}/{}".format(frames_dir, cat)
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    command = ["tar", "-xzf", tar_gz_file, "-C", target_dir]
    subprocess.call(command)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "yes":
        print("Confirm by providing 'yes' as argument")
        sys.exit(1)

    train_files_to_unzip = glob.glob("{}/train/*.tar.gz".format(zipped_frames_dir))
    val_files_to_unzip = glob.glob("{}/val/*.tar.gz".format(zipped_frames_dir))

    files_to_unzip = []
    for file in train_files_to_unzip:
        files_to_unzip.append([file, "train"])
    for file in val_files_to_unzip:
        files_to_unzip.append([file, "val"])
    
    for file, cat in tqdm.tqdm(files_to_unzip):
        unzip(file, cat)
