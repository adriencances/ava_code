import sys
import glob
import tqdm


frames_dir = "/media/hdd/adrien/Ava_v2.2/correct_frames"
output_file = "/home/acances/Code/ava/frames_by_timestamp.csv"


def compute_frames_by_timestamp(video_folder, frames_by_timestamp_list):
    video_id = video_folder.split("/")[-1]
    timestamp_folders = glob.glob("{}/*".format(video_folder))
    N = len(glob.glob("{}/*".format(timestamp_folders[0])))
    for timestamp_folder in timestamp_folders:
        assert len(glob.glob("{}/*".format(timestamp_folder))) == N
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

    video_folders = glob.glob("{}/train/*".format(frames_dir))
    video_folders += glob.glob("{}/val/*".format(frames_dir))
    frames_by_timestamp_list = []
    for video_folder in tqdm.tqdm(video_folders):
        compute_frames_by_timestamp(video_folder, frames_by_timestamp_list)
    
    write_output(output_file, frames_by_timestamp_list)
