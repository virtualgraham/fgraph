stride = 16
window_size = 32
playback_random_walk_length = 10

video_path = "../../media/Tabletop Objects/videos"
db_path = "../../data/table_top_objects_10.db"
patch_dir = "../../patches/table_top_objects_10"

from vgg16_window_walker_lib_b import build_graph

build_graph(db_path, video_path, patch_dir, stride=stride, window_size=window_size, max_files=10)
