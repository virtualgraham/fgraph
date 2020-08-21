stride = 16
window_size = 32
playback_random_walk_length = 10


video_path = "../../media/tabletop_objects/videos/"
db_path = "../../data/table_objects_80.db"

# video_path = "../../media/tabletop_objects/videos/288_brush_carrot_clippers_cup_flowers_hanger_ketchup.mp4"
# db_path = "../../data/test1.db"

from vgg16_window_walker_lib_c import build_graph

build_graph(db_path, video_path, stride=stride, window_size=window_size, max_files=80, keep_times=False, max_elements=10000000)
