import glob
import sys


pairs_dir = "/home/acances/Data/Ava_v2.2/pairs"


def nb_lines(file):
    with open(file, "r") as f:
        for i, line in enumerate(f): pass
    return i + 1


if __name__ == "__main__":
    cat = sys.argv[1]
    pair_type = sys.argv[2]
    
    positive_pairs_files = glob.glob("{}/{}/{}/*".format(pairs_dir, cat, pair_type))

    positive_pairs = {}
    for file in positive_pairs_files:
        video_id = file.split("/")[-1].split(".")[0][6:]
        positive_pairs[video_id] = nb_lines(file)
    
    min_positive_pairs = min(positive_pairs.values())
    max_positive_pairs = max(positive_pairs.values())

    total_positive_pairs = sum(positive_pairs.values())
    nb_videos = len(positive_pairs)

    # print("\t".join(map(str, sorted(positive_pairs.values()))))

    print("Total:\t{}".format(total_positive_pairs))
    print("Mean by file:\t{}".format(total_positive_pairs/nb_videos))
    print("Min:\t{}".format(min_positive_pairs))
    print("Max:\t{}".format(max_positive_pairs))
