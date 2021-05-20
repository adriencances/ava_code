from generate_positive_and_medium_negative_pairs import compute_pos_and_medneg_pairs
from generate_hard_negative_pairs import compute_hard_negative_pairs
from generate_easy_negative_pairs import compute_easy_negative_pairs

from count_pairs import get_nb_pairs


def generate_all_pairs():
    compute_pos_and_medneg_pairs()
    compute_hard_negative_pairs()
    compute_easy_negative_pairs()


if __name__ == "__main__":
    generate_all_pairs()
