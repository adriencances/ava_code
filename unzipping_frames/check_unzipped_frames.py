import subprocess
import sys
import glob
import tqdm


frames_dir = "/media/hdd/adrien/Ava_v2.2/frames"


def check(video_folder):
    subfolders = glob.glob(video_folder + "/*")
    for subfolder in subfolders:
        content = glob.glob(subfolder + "/*")
        assert len(content) > 0


if __name__ == "__main__":
    train_video_folders_to_check = glob.glob("{}/train/*".format(frames_dir))
    val_video_folders_to_check = glob.glob("{}/val/*".format(frames_dir))

    assert len(train_video_folders_to_check) == 235
    assert len(val_video_folders_to_check) == 64

    video_folders_to_check = train_video_folders_to_check + val_video_folders_to_check
    for video_folder in tqdm.tqdm(video_folders_to_check):
        check(video_folder)
