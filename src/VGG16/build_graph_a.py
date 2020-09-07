stride = 16
window_size = 32
center_size = 16


video_path = "../../media/tabletop_objects/videos/"
mask_path = "../../media/tabletop_objects/masks/"
db_path = "../../data/table_objects_h.db"

video_files = [
	'333_apple_bear_carrot_chain_clippers_cup_notebook_opener.mp4', 
	'360_brush_chain_cup_hanger_ketchup_opener_pepper_rock.mp4', 
	'337_apple_carrot_clippers_cologne_cup_hanger_notebook_pepper.mp4', 
	'335_chain_cologne_cup_hanger_ketchup_notebook_opener_shorts.mp4', 
	'344_brush_chain_clippers_cologne_cup_flowers_hanger_pepper.mp4', 
	'361_apple_bear_brush_cologne_flowers_notebook_rock_shorts.mp4', 
	'336_apple_carrot_clippers_cologne_hanger_notebook_pepper_rock.mp4',
	'339_bear_chain_clippers_flowers_hanger_notebook_opener_pepper.mp4',
	'325_bear_brush_carrot_clippers_cologne_ketchup_pepper_shorts.mp4', 
	'359_apple_bear_carrot_flowers_ketchup_opener_rock_shorts.mp4', 
	'365_bear_brush_carrot_chain_cup_flowers_ketchup_shorts.mp4', 
	'370_apple_brush_flowers_hanger_ketchup_opener_rock_shorts.mp4'
] 

from vgg16_window_walker_lib_h import build_graph

build_graph(db_path, video_path, mask_path, video_files,  stride=stride, window_size=window_size, center_size=center_size, keep_times=False, max_elements=1200000)
