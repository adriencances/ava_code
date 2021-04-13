import glob
import os
import tqdm


frames_dir = "/media/hdd/adrien/Ava_v2.2/frames"

train_GT_file = "/home/acances/Data/Ava_v2.2/ava_train_v2.2.csv"
val_GT_file = "/home/acances/Data/Ava_v2.2/ava_val_v2.2.csv"


def check(cat, GT_file):
    nb_lines = len([1 for line in open(GT_file, "r")])
    with open(GT_file, "r") as f:
        for line in tqdm.tqdm(f, total=nb_lines):
            video_id, timestamp = line.strip().split(",")[:2]
            timestamp = int(timestamp)

            folder = "{}/{}/{}/{:05d}".format(frames_dir, cat, video_id, timestamp)
            assert os.path.isdir(folder)
                # print("Missing timestamp:\t{}_{:05d}".format(video_id, timestamp))

            content = glob.glob("{}/*".format(folder))
            assert len(content) > 0
                # print("Empty folder:\t{}_{:05d}".format(video_id, timestamp))


if __name__ == "__main__":
    check("train", train_GT_file)
    check("val", val_GT_file)
