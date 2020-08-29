import numpy as np
import cv2
from os.path import join
import random



max_pos_attempts = 1000
mask_path = "./media/tabletop_objects/masks/"
video_path = "./media/tabletop_objects/videos/"
db_path = "./data/table_objects_10.db"
max_frames = 30*30
walker_count = 10
window_size = 32
stride = 16
center_size = 16
center_threshold = center_size*center_size*0.9


videos = [
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
   

def extract_window(frame, pos, window_size):
    half_w = window_size/2.0
    bottom_left = [int(round(pos[0]-half_w)), int(round(pos[1]-half_w))]
    top_right = [bottom_left[0]+window_size, bottom_left[1]+window_size]
   
    if bottom_left[0] < 0:
        top_right[0] -= bottom_left[0]
        bottom_left[0] = 0

    if bottom_left[1] < 0:
        top_right[1] -= bottom_left[1]
        bottom_left[1] = 0

    if top_right[0] >= frame.shape[0]:
        bottom_left[0] -= (top_right[0]-frame.shape[0]+1)
        top_right[0] = frame.shape[0]-1

    if top_right[1] >= frame.shape[1]:
        bottom_left[1] -= (top_right[1]-frame.shape[1]+1)
        top_right[1] = frame.shape[1]-1

    return frame[bottom_left[0]:top_right[0], bottom_left[1]:top_right[1]]



def is_window_on_object(pos, mask_frame):
    # np.nonzero(x)

    w_half = window_size/2.0
    w_bottom_left = [int(round(pos[0]-w_half)), int(round(pos[1]-w_half))]
    w_top_right = [w_bottom_left[0]+window_size, w_bottom_left[1]+window_size]

    if w_bottom_left[0] < 0:
        return False

    if w_bottom_left[1] < 0:
        return False

    if w_top_right[0] >= mask_frame.shape[0]:
        return False

    if w_top_right[1] >= mask_frame.shape[1]:
        return False

    c_half = center_size/2
    c_bottom_left = [int(round(pos[0]-c_half)), int(round(pos[1]-c_half))]
    c_top_right = [c_bottom_left[0]+center_size, c_bottom_left[1]+center_size]

    center = mask_frame[c_bottom_left[0]:c_top_right[0], c_bottom_left[1]:c_top_right[1]]

    return np.sum(center.any(axis=2)) >= center_threshold


def first_pos(mask_frame):
    for _ in range(max_pos_attempts):
        pos = (mask_frame.shape[0] * random.random(),  mask_frame.shape[1] * random.random())
        if is_window_on_object(pos, mask_frame):
            return pos
    return None


def next_pos(mask_frame, pos):
    if pos is None:
        return first_pos(mask_frame)

    n = [(pos[0] - stride, pos[1]), 
        (pos[0] + stride, pos[1]), 
        (pos[0], pos[1] - stride), 
        (pos[0], pos[1] + stride), 
        (pos[0] + stride, pos[1] + stride), 
        (pos[0] - stride, pos[1] + stride), 
        (pos[0] + stride, pos[1] - stride), 
        (pos[0] - stride, pos[1] - stride)]

    random.shuffle(n)

    for p in n:
        if is_window_on_object(p,mask_frame):
            return p

    return None




color_dist = 15


color_vectors = [
    np.array((0,0,0)),
    np.array((77,198,69)),
    np.array((255,255,255)),
    np.array((32,233,249)),
    np.array((54,123,235)),
    np.array((173,7,135)),
    np.array((110,0,0)),
    np.array((203,252,254)),
    np.array((211,180,242)),
    np.array((224,17,224)),
    np.array((240,222,78)),
    np.array((0,0,115)),
    np.array((195,255,176)),
    np.array((168,168,168)),
    np.array((74,0,212)),
    np.array((74,254,193)),
    np.array((178,213,251)), 
]


color_full = np.array([np.tile(cv, (1280, 720, 1)) for cv in color_vectors])


color_objects = {
    None,
    "apple",
    "bear",
    "brush",
    "carrot",
    "chain",
    "clippers",
    "cologne",
    "cup",
    "flowers",
    "hanger",
    "ketchup",
    "notebook",
    "opener",
    "pepper",
    "rock",
    "shorts",
}


def color_fun(mask_frame):
    o = np.zeros((1280, 720), dtype=np.uint8)

    for i in range(1, len(color_vectors)):
        o[np.linalg.norm(color_full[i] - mask_frame, axis=2)<color_dist] = i
    
    return o





def run(file):

    mask = cv2.VideoCapture(join(mask_path,"mask_"+file))
    video = cv2.VideoCapture(join(video_path,file))

    done = False
    pos = None

    mask_ret, mask_frame = mask.read()
    video_ret, video_frame = video.read()

    if mask_ret == False or video_ret == False:
        done = True
        return

    # pos = next_pos(mask_frame, pos)
    # window = extract_window(mask_frame, pos, window_size)
    # print(mask_frame.shape, mask_frame.dtype)

    # # d = np.apply_along_axis(color_fun, 2, mask_frame)

    o = color_fun(mask_frame)
    unique, counts = np.unique(o, return_counts=True)
    print(dict(zip(unique, counts)))

    # cv2.imshow('window', window)

    # cv2.waitKey(5000)


        

# colors = {
#     (178  19 142): "chain", 
# }


# t1 = np.tile(a1, (32,32,1))
# a2 = np.array((184,17,145))
# t2 = np.tile(a2, (32,32,1))
# np.linalg.norm(t1-t2)
# np.linalg.norm(t1-t2, axis=2)

np.set_printoptions(threshold=np.inf)

run("325_bear_brush_carrot_clippers_cologne_ketchup_pepper_shorts.mp4")

