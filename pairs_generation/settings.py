class Settings:
    SEGMENT_LENGTH = 16
    TEMP_INTERSECTION_THRESHOLD = SEGMENT_LENGTH
    IOU_THRESHOLD = 0.2
    FRAME_PROPORTION = 0.4
    MAX_PEOPLE_IN_SHOT = 4

    SHIFT = SEGMENT_LENGTH // 2

    tracks_dir = "/home/acances/Data/Ava_v2.2/tracks"
    pairs_dir = "/home/acances/Data/Ava_v2.2/pairs16_new/"
