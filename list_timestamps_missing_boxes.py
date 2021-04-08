import pickle
import numpy as np
import sys
import glob




interesting_timestamps = []
if __name__ == "__main__":
    video_id = sys.argv[1]
    tracks_path = "/home/acances/Data/Ava_v2.2/tracks/train/{}/".format(video_id)
    tracks_files = sorted(glob.glob(tracks_path + "*"))
    for tracks_file in tracks_files:
        timestamp = tracks_file.split("/")[-1].split("_")[0]

        with open(tracks_file, "rb") as f: data = pickle.load(f)
        min_id = min([track[0,0] for track, _ in data]) if data != [] else 25
        max_id = max([track[-1,0] for track, _ in data]) if data != [] else 0
        if data == []:
            print("Tracks for {} empty".format(timestamp))

        # if min_id > 1 or max_id < 25:
        # if max_id - min_id + 1 < 25:
        if min_id == 1:
            interesting_timestamps.append(timestamp)

        # if max_id == 25 and min_id == 0:
        #     for track, score in data:
        #         if track[0][0] != 0 or len(track) < 2:
        #             continue
        #         if np.sum(np.abs(track[0][1:5] - track[1][1:5])) != 0:
        #             interesting_timestamps.append(timestamp)
        #             break

    print("\t".join(interesting_timestamps))
