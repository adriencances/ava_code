import glob
import sys

from settings import Settings


pairs_dir = Settings.pairs_dir


def nb_lines(file):
    i = -1
    with open(file, "r") as f:
        for i, line in enumerate(f): pass
    return i + 1


def get_nb_pairs(cat, pair_type):
    pairs_files = glob.glob("{}/{}/{}/*".format(pairs_dir, cat, pair_type))

    nb_pairs = 0
    for file in pairs_files:
        nb_pairs += nb_lines(file)
    
    return nb_pairs


def print_general_stats():
    pair_types = ["positive", "hard_negative", "medium_negative", "easy_negative"]
    cats = ["train", "val"]
    print("\t".join(pair_types))
    for cat in cats:
        numbers = [get_nb_pairs(cat, pair_type) for pair_type in pair_types]
        nb_positives = numbers[0]
        percentages = [round(nb / nb_positives * 100, 1) for nb in numbers]
        print(cat)
        print("total:\t" + str(sum(numbers)))
        print("\t".join(map(str, percentages)))
        print("\t".join(map(str, numbers)))


def print_stats(cat, pair_type):
    pairs_files = glob.glob("{}/{}/{}/*".format(pairs_dir, cat, pair_type))

    pairs = {}
    for file in pairs_files:
        video_id = file.split("/")[-1].split(".")[0][6:]
        pairs[video_id] = nb_lines(file)
    
    min_pairs = min(pairs.values())
    max_pairs = max(pairs.values())

    total_pairs = sum(pairs.values())
    nb_videos = len(pairs)

    print("Total:\t{}".format(total_pairs))
    print("Mean by file:\t{}".format(total_pairs/nb_videos))
    print("Min:\t{}".format(min_pairs))
    print("Max:\t{}".format(max_pairs))


if __name__ == "__main__":
    if len(sys.argv) > 2:
        cat = sys.argv[1]
        pair_type = sys.argv[2]
        print_stats(cat, pair_type)
    else:
        print_general_stats()
