import cv2
import pickle
import glob
import sys
import os
from pathlib import Path


frames_root_dir = "/home/acances/Data/Ava_v2.2/train_videos/frames"

tracks_root_dir = "/home/acances/Data/Ava_v2.2/normalized_tracks/train"

# target_root_dir = "/home/acances/Code/ava/annotated_frames"
target_root_dir = "/home/acances/Code/ava/new_annotated_frames_middle"


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


def get_tracks(tracks_file):
    with open(tracks_file, "rb") as f:
        tracks = pickle.load(f)
    return tracks


def annotate_timestamp(video_id, timestamp):
    frames_folder = "{}/{}/{:05d}".format(frames_root_dir, video_id, timestamp)
    frames_files = sorted(glob.glob(frames_folder + "/*"))
    tracks_file = "{}/{}/{:05d}_tracks.pkl".format(tracks_root_dir, video_id, timestamp)
    next_tracks_file = "{}/{}/{:05d}_tracks.pkl".format(tracks_root_dir, video_id, timestamp + 1)
    if not os.path.isfile(tracks_file):
        print("Tracks file does not exist for timestamp {:05d} of video {}".format(timestamp, video_id))
        return
    if not os.path.isfile(next_tracks_file):
        print("Tracks file does not exist for timestamp {:05d} of video {}".format(timestamp + 1, video_id))
        return

    target_directory = "{}/{}/{:05d}".format(target_root_dir, video_id, timestamp)
    Path(target_directory).mkdir(parents=True, exist_ok=True)

    tracks = get_tracks(tracks_file)
    next_tracks = get_tracks(next_tracks_file)

    nb_frames = len(frames_files)

    boxes_by_frame = dict([(frame_id, []) for frame_id in range(nb_frames + 1)])
    for track in tracks:
        for box_info in track[0]:
            frame_id = int(box_info[0])
            box = list(box_info[1:5])
            boxes_by_frame[frame_id].append(box)
    
    next_boxes_by_frame = dict([(frame_id, []) for frame_id in range(nb_frames + 1)])
    for track in next_tracks:
        for box_info in track[0]:
            frame_id = int(box_info[0])
            box = list(box_info[1:5])
            next_boxes_by_frame[frame_id].append(box)

    for frame_id in range(1, nb_frames//2):
        new_frame_file = "{}/{:05d}.jpg".format(target_directory, frame_id)
        make_annotated_frame(frames_files[frame_id], new_frame_file, boxes_by_frame[frame_id + nb_frames//2 + 1])
        if boxes_by_frame[frame_id + nb_frames//2 + 1] == []:
            print(frame_id)

    
    for frame_id in range(nb_frames//2, nb_frames):
        new_frame_file = "{}/{:05d}.jpg".format(target_directory, frame_id)
        make_annotated_frame(frames_files[frame_id], new_frame_file, next_boxes_by_frame[frame_id + nb_frames//2 + 1 - nb_frames])
        if next_boxes_by_frame[frame_id + nb_frames//2 + 1 - nb_frames] == []:
            print(frame_id)
    
    print("nb_frames: ", nb_frames)


if __name__ == "__main__":
    video_id = sys.argv[1]
    timestamp = int(sys.argv[2])
    annotate_timestamp(video_id, timestamp)




