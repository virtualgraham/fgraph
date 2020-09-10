video_path = "../../media/tabletop_objects/videos/"
mask_path = "../../media/tabletop_objects/masks/"
db_path = "../../data/table_objects_i.db"

params = {
	"runs": 1,
    "window_size": 32, 
	"grid_margin": 16, 
    "max_frames": 30*30,
	"search_max_frames": 30, 
	"max_elements": 12000000,
    "space": 'cosine', 
    "dim": 512, 
    "ef": 300, 
    "M": 64, 
	"rebuild_index": False,
    "keep_times": False,

	"stride": 24,
	"center_size": 16,
	"walk_length": 100,
    "walker_count": 500,
    "prevent_similar_adjacencies": False,
    "knn": 50,
    "accurate_prediction_limit": 12,
    "distance_threshold": 0.15,
    "prediction_history_length": 7,
    "history_community_matches": 1,
    "identical_distance": 0.01,
    "search_walker_count": 4,
    "search_walk_length": 10,
    "feature_dis": 0.4,
    "community_dis": 0.200,
    "search_knn": 100,
    "initial_walk_length": 8,  
    "member_portion": 100,
    "walk_trials": 1000,
}

video_files = [
	'333_apple_bear_carrot_chain_clippers_cup_notebook_opener.mp4', 
	# '360_brush_chain_cup_hanger_ketchup_opener_pepper_rock.mp4', 
	# '337_apple_carrot_clippers_cologne_cup_hanger_notebook_pepper.mp4', 
	# '335_chain_cologne_cup_hanger_ketchup_notebook_opener_shorts.mp4', 
	# '344_brush_chain_clippers_cologne_cup_flowers_hanger_pepper.mp4', 
	# '361_apple_bear_brush_cologne_flowers_notebook_rock_shorts.mp4', 
	# '336_apple_carrot_clippers_cologne_hanger_notebook_pepper_rock.mp4',
	# '339_bear_chain_clippers_flowers_hanger_notebook_opener_pepper.mp4',
	# '325_bear_brush_carrot_clippers_cologne_ketchup_pepper_shorts.mp4', 
	# '359_apple_bear_carrot_flowers_ketchup_opener_rock_shorts.mp4', 
	# '365_bear_brush_carrot_chain_cup_flowers_ketchup_shorts.mp4', 
	# '370_apple_brush_flowers_hanger_ketchup_opener_rock_shorts.mp4'
] 

from vgg16_window_walker_lib_i import build_graph

build_graph(db_path, video_path, mask_path, video_files, params)
