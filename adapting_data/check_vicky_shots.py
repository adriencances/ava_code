import glob
import sys
import tqdm


shots_dir = "/home/acances/Data/Ava_v2.2/final_shots"


def check_shot_file(file):
    video_id = file.split("/")[-1].split(".")[0][6:]
    boundaries = []
    with open(file, "r") as f:
        for line in f:
            start, end = line.strip().split(",")
            t1, n1 = tuple(map(int, start.split("_")))
            t2, n2 = tuple(map(int, end.split("_")))
            boundaries.append([(t1, n1), (t2, n2)])
    # assert boundaries[0][0][1] == 0
    if boundaries[0][0][1] == 0:
        print("LA")
    assert boundaries[0][1][1] != 0
    for i in range(1, len(boundaries)):
        assert boundaries[i][0][1] != 0
        assert boundaries[i][1][1] != 0
    assert boundaries[0][0][0] >= 902
    assert boundaries[-1][1][0] <= 1798
    # for i in range(len(boundaries) - 1):
    #     assert boundaries[i][1] == boundaries[i + 1][0]
    max_n = max([bds[0][1] for bds in boundaries])
    last_n = boundaries[-1][1][1]
    if last_n not in [max_n, max_n - 1]:
        print("ATTENTION")
    N = boundaries[-1][1][1]
    # print(N)


if __name__ == "__main__":
    shots_files = glob.glob("{}/train/*".format(shots_dir))
    shots_files += glob.glob("{}/val/*".format(shots_dir))
    for file in tqdm.tqdm(shots_files):
        check_shot_file(file)
