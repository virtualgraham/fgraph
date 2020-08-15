stride = 16
window_size = 32
playback_random_walk_length = 10

video_path = "../../media/Tabletop Objects/videos/001_apple.mp4"
db_path = "../../data/test_001_apple.db"
patch_dir = "../../patches"

from vgg16_window_walker_lib_a import build_graph

build_graph(db_path, video_path, patch_dir, stride=stride, window_size=window_size)