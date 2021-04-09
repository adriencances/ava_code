import cv2
import pickle
import glob
import sys
import os
from pathlib import Path


# frames_root_dir = "/home/acances/Data/Ava_v2.2/train_videos/frames"
frames_root_dir = "/home/acances/Data/Ava_v2.2"

bboxes_root_dir = "/home/acances/Data/Ava_v2.2/bboxes_by_keyframes/train"

# target_root_dir = "/home/acances/Code/ava/keyframes_annotated_first/train"
target_root_dir = "/home/acances/Code/ava/new_keyframes_annotated_first/train"


def make_annotated_frame(frame_file, new_frame_file, boxes):
    img = cv2.imread(frame_file)
    height, width = img.shape[:2]
    color = (0, 255, 0)
    thickness = 2
    for box in boxes:
        box[0] *= width
        box[1] *= height
        box[2] *= width
        box[3] *= height
        box = list(map(int, list(box)))
        img = cv2.rectangle(img, tuple(box[:2]), tuple(box[2:]), color, thickness)
    cv2.imwrite(new_frame_file, img)


def get_boxes(bboxes_file):
    boxes = []
    with open(bboxes_file, "r") as f:
        for line in f:
            box = list(map(float, line.strip().split(",")))[:-1]
            boxes.append(box)
    return boxes


def annotate_timestamp(video_id, timestamp):
    frames_folder = "{}/{}/{:05d}".format(frames_root_dir, video_id, timestamp)

    frames_files = sorted(glob.glob(frames_folder + "/*"))
    nb_frames = len(frames_files)
    keyframe_file = frames_files[nb_frames//2]
    # keyframe_file = frames_files[0]

    bboxes_file = "{}/{}/{:05d}_bboxes.csv".format(bboxes_root_dir, video_id, timestamp)
    if not os.path.isfile(bboxes_file):
        print("Bboxes file does not exist for timestamp {:05d} of video {}".format(timestamp, video_id))
        return


    Path("{}/{}".format(target_root_dir, video_id)).mkdir(parents=True, exist_ok=True)
    new_keyframe_file = "{}/{}/{:05d}.jpg".format(target_root_dir, video_id, timestamp)

    boxes = get_boxes(bboxes_file)
    make_annotated_frame(keyframe_file, new_keyframe_file, boxes)


if __name__ == "__main__":
    video_id = sys.argv[1]
    timestamp = int(sys.argv[2])
    annotate_timestamp(video_id, timestamp)




