import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math as m
from torchvision import transforms, utils
import cv2
import sys
import pickle
import tqdm


frames_dir = "/media/hdd/adrien/Ava_v2.2/correct_frames"
shots_dir = "/home/acances/Data/Ava_v2.2/final_shots"
tracks_dir = "/home/acances/Data/Ava_v2.2/tracks"
fbt_file = "/home/acances/Code/ava/frames_by_timestamp.csv"


class FrameProcessor:
    def __init__(self):
        self.nb_frames_by_timestamp = {}
        with open(fbt_file, "r") as f:
            for line in f:
                vid_id, N = line.strip().split(",")
                self.nb_frames_by_timestamp[video_id] = int(N)


def get_boxes(bboxes_file):
    boxes = []
    with open(bboxes_file, "r") as f:
        for line in f:
            box = list(map(float, line.strip().split(",")))[:-1]
            boxes.append(box)
    return boxes


def get_enlarged_box(box, alpha):
    # Enlarge the box area by 100*alpha percent while preserving
    # the center and the aspect ratio
    beta = 1 + alpha
    x1, y1, x2, y2 = box
    dx = x2 - x1
    dy = y2 - y1
    x1 -= (np.sqrt(beta) - 1)*dx/2
    x2 += (np.sqrt(beta) - 1)*dx/2
    y1 -= (np.sqrt(beta) - 1)*dy/2
    y2 += (np.sqrt(beta) - 1)*dy/2
    return x1, y1, x2, y2


def get_preprocessed_frame(video_id, cat, t, n):
    # t : timestamp index of the video
    # n : frame index in the timestamp (frame indices start at 1)
    frame_file = "{}/{}/{}/{:05d}/{:06d}.jpg".format(frames_dir, cat, video_id, t, n)
    # frame : H * W * 3
    frame = cv2.imread(frame_file)
    # frame : 3 * W * H
    frame = frame.transpose(2, 1, 0)
    frame = torch.from_numpy(frame)
    return frame


def get_processed_frame(frame, box, w, h, normalized_box=False):
    # frame : 3 * W * H
    # (w, h) : dimensions of new frame

    C, W, H = frame.shape
    x1, y1, x2, y2 = box

    # If box is in normalized coords, i.e.
    # image top-left corner (0,0), bottom-right (1, 1),
    # then turn normalized coord into absolute coords
    if normalized_box:
        x1 = x1*W
        x2 = x2*W
        y1 = y1*H
        y2 = y2*H

    # Round coords to integers
    X1 = max(0, m.floor(x1))
    X2 = max(0, m.ceil(x2))
    Y1 = max(0, m.floor(y1))
    Y2 = max(0, m.ceil(y2))
    
    dX = X2 - X1
    dY = Y2 - Y1

    # Get the cropped bounding box
    boxed_frame = transforms.functional.crop(frame, X1, Y1, dX, dY)
    dX, dY = boxed_frame.shape[1:]

    # Compute size to resize the cropped bounding box to
    if dY/dX >= h/w:
        w_tild = m.floor(dX/dY*h)
        h_tild = h
    else:
        w_tild = w
        h_tild = m.floor(dY/dX*w)
    assert w_tild <= w
    assert h_tild <= h

    # Get the resized cropped bounding box
    resized_boxed_frame = transforms.functional.resize(boxed_frame, [w_tild, h_tild])

    # Put the resized cropped bounding box on a gray canvas
    new_frame = 127*torch.ones(C, w, h)
    i = m.floor((w - w_tild)/2)
    j = m.floor((h - h_tild)/2)
    new_frame[:, i:i+w_tild, j:j+h_tild] = resized_boxed_frame

    return new_frame


def nb_frames_per_timestamp(video_id):
    with open(fbt_file, "r") as f:
        for line in f:
            vid_id, N = line.strip().split(",")
            if video_id == vid_id:
                return int(N)
    print("WARNING: no information for video id {} in fbt_file".format(video_id))
    return None


def get_tracks(video_id, cat, shot_id):
    tracks_file = "{}/{}/{}/{:05d}_tracks.pkl".format(tracks_dir, cat, video_id, shot_id)
    with open(tracks_file, "rb") as f:
        tracks = pickle.load(f)
    return tracks


def get_extreme_timestamps(video_id, cat, shot_id):
    shots_file = "{}/{}/shots_{}.csv".format(shots_dir, cat, video_id)
    with open(shots_file, "r") as f:
        for i, line in enumerate(f):
            if i == shot_id:
                start, end = line.strip().split(",")
                t1, n1 = tuple(map(int, start.split("_")))
                t2, n2 = tuple(map(int, end.split("_")))
                return t1, t2
    print("WARNING: no shot of index {} for video {}".format(shot_id, video_id))
    return None


def get_processed_track_frames(video_id, cat, track, t1, t2, begin_frame, end_frame, w, h, alpha, normalized_box=False):
    # begin_frame, end_frame : indices in [0, (t2-t1+1)N - 1]
    # t1, t2 : first and last timestamps (included) corresponding to the shot to which the track belongs
    N = nb_frames_per_timestamp(video_id)
    b = int(track[0, 0])
    processed_frames = []
    for i in range(begin_frame, end_frame):
        t = t1 + i//N
        n = i%N + 1
        frame = get_preprocessed_frame(video_id, cat, t, n)
        track_frame_index = i - b
        box = track[track_frame_index][1:5]
        box = get_enlarged_box(box, alpha)
        processed_frame = get_processed_frame(frame, box, w, h, normalized_box)
        processed_frames.append(processed_frame)
    processed_frames = torch.stack(processed_frames, dim=0)
    return processed_frames


def get_frames(video_id, cat, shot_id, track_id, begin_frame, end_frame):
    # shot_id : 0-based index.
    # track_id : 0-based index.
    # begin_frame, end_frame : indices between 0 and (t2-t1+1)N - 1,
    # where t1 and t2 are the first and last (included) timestamps for the considered shot,
    # and where N is the number of frames per timestamp for the considered video.
    # Warning: end_frame is the index of the first frame not included

    # Use dictionary such that d[video_id][shot_id] = (t1, t2)
    t1, t2 = None, None
    # Use dictionary such that d[video_id] = N
    N = None
    frames = []
    for i in range(begin_frame, end_frame):
        t = t1 + i//N
        n = i%N + 1
        frame = get_preprocessed_frame(video_id, cat, t, n)
        frames.append(frame)

    tracks_file = "{}/{}/{}/{:05d}_tracks.pkl ".format(tracks_dir, cat, video_id, shot_id)
    tracks = get_tracks(video_id, cat, shot_id)
    track, score = tracks[track_id]
    b = int(track[0, 0])
    boxes = track[begin_frame - b:ending_frame - b, 1:5]

    assert len(boxes) == len(frames)
    
    
def print_out_processed_frames(processed_frames):
    target_dir = "/home/acances/Code/ava/various"
    nb_frames = processed_frames.shape[0]
    for i in range(nb_frames):
        frame = processed_frames[i].numpy().transpose(2, 1, 0)
        target_file = "{}/{:05d}.jpg".format(target_dir, i + 1)
        cv2.imwrite(target_file, frame)


if __name__ == "__main__":
    tracks_file = sys.argv[1]

    shot_id = int(tracks_file.split("/")[-1].split("_")[0])
    video_id = tracks_file.split("/")[-2]
    cat = tracks_file.split("/")[-3]

    tracks = get_tracks(video_id, cat, shot_id)
    t1, t2 = get_extreme_timestamps(video_id, cat, shot_id)
    track, score = tracks[0]
    begin_frame = int(track[0, 0])
    end_frame = int(track[-1, 0])

    w, h = 224, 224
    alpha = 0.1

    processed_frames = get_processed_track_frames(video_id, cat, track, t1, t2, begin_frame, end_frame, w, h, alpha)
    print_out_processed_frames(processed_frames)
    print(processed_frames.shape)
