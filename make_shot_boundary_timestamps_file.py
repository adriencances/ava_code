import sys
import glob
import tqdm


shots_dir = "/media/hdd/adrien/Ava_v2.2/final_shots"
output_file = "/home/acances/Code/ava/boundary_timestamps.csv"


def get_boundary_timestamps(shot_file, boundary_timestamps_list):
    video_id = shot_file.split("/")[-1].split(".")[0][6:]
    with open(shot_file, "r") as f:
        for line in f:
            start, end = line.strip().split(",")
            t1, n1 = tuple(map(int, start.split("_")))
            t2, n2 = tuple(map(int, end.split("_")))
            
    frames_by_timestamp_list.append([video_id, N])


def write_output(output_file, frames_by_timestamp_list):
    with open(output_file, "w") as f:
        for video_id, N in frames_by_timestamp_list:
            f.write("{},{}\n".format(video_id, str(N)))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Confirm by providing 'yes' as argument")
        sys.exit(1)
    confirm = sys.argv[1]
    if confirm != "yes":
        print("Confirm by providing 'yes' as argument")
        sys.exit(1)

    shots_files = glob.glob("{}/train/*".format(frames_dir))
    shots_files += glob.glob("{}/val/*".format(frames_dir))
    frames_by_timestamp_list = []
    for shot_file in tqdm.tqdm(shots_files):
        compute_frames_by_timestamp(shot_file, frames_by_timestamp_list)
    
    write_output(output_file, frames_by_timestamp_list)
