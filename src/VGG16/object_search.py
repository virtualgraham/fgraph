import numpy as np
import cv2
from os.path import join
import random
import math 

from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.vgg16 import preprocess_input

from vgg16_window_walker_lib_d import color_fun, extract_windows, extract_window, extract_object, get_rad_grid, paint_windows, MemoryGraph


max_pos_attempts = 1000
mask_path = "../../media/tabletop_objects/masks/"
video_path = "../../media/tabletop_objects/videos/"
db_path = "../../data/table_objects_mask.db"
max_frames = 30*30
walker_count = 3
window_size = 32
stride = 16
center_size = 16
center_threshold = center_size*center_size*0.9
grid_margin = 16
walk_length = 7
max_elements=1000000

def key_point_grid(orb, frame, obj_frame, stride):

    grid_width = math.floor((frame.shape[0] - grid_margin*2) / stride)
    grid_height = math.floor((frame.shape[1] - grid_margin*2) / stride)

    grid_offset_x = ((frame.shape[0] - grid_margin*2) % stride)/2.0 + grid_margin
    grid_offset_y = ((frame.shape[1] - grid_margin*2) % stride)/2.0 + grid_margin

    object_grid_locations = set()

    print("grid_width", grid_width, "grid_height", grid_height)
    for x in range(grid_width):
        for y in range(grid_height):
            p = (grid_offset_x + x * stride + 0.5 * stride, grid_offset_y + y * stride + 0.5 * stride)
            w = extract_window(obj_frame, p, stride)
            if extract_object(w, stride) is not None:
                object_grid_locations.add((x, y))
    
    print("len(object_grid_locations)", len(object_grid_locations))
    kp = orb.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
    print("len(kp)", len(kp))

    grid = dict()

    for k in kp:
        p = (k.pt[1],k.pt[0])
        g = (int(math.floor((p[0]-grid_offset_x)/stride)), int(math.floor((p[1]-grid_offset_y)/stride)))
        if g in object_grid_locations:
            if g in grid:
                grid[g].append(p)
            else:
                grid[g] = [p]

    return grid


def first_pos(kp_grid):
    loc = random.choice(list(kp_grid.keys()))
    return loc, random.choice(kp_grid[loc])


def next_pos(kp_grid, shape, g_pos, walk_t, walk_length, stride):
 
    if (g_pos is not None) and walk_t < walk_length:

        for rad in range(1, 3):
            rad_grid = get_rad_grid(g_pos, rad, shape, stride)

            if len(rad_grid) == 0:
                print("frame empty?")
                break

            random.shuffle(rad_grid)

            for loc in rad_grid:
                if loc in kp_grid:
                    return loc, random.choice(kp_grid[loc]), True
    
    loc, pos = first_pos(kp_grid)
    return loc, pos, False


def run(file):

    # memory_graph = MemoryGraph(db_path, space='cosine', dim=512, max_elements=max_elements)
    cnn = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(32, 32, 3))
    orb = cv2.ORB_create(nfeatures=100000, fastThreshold=7)

    mask = cv2.VideoCapture(join(mask_path,"mask_"+file))
    video = cv2.VideoCapture(join(video_path,file))

    g_pos = [None for _ in range(walker_count)]
    pos = [None for _ in range(walker_count)]
    adj = [False for _ in range(walker_count)]
    walk_t = [0 for _ in range(walker_count)]
    cluster_feats = [[] for _ in range(walker_count)]
    cluster_positions = [[] for _ in range(walker_count)]
    cluster_patches = [[] for _ in range(walker_count)]

    for t in range(max_frames):
        mask_ret, mask_frame = mask.read()
        video_ret, video_frame = video.read()

        if mask_ret == False or video_ret == False:
            adj = [False for _ in range(walker_count)]
        else:
            video_shape = video_frame.shape

            obj_frame = color_fun(mask_frame)
            kp_grid = key_point_grid(orb, video_frame, obj_frame, stride)

            for i in range(walker_count):
                g_pos[i], pos[i], adj[i] = next_pos(kp_grid, video_shape, g_pos[i], walk_t[i], walk_length, stride)
                if t == 0:
                    adj[i] = True
                if adj[i]:
                    walk_t[i] += 1
                else:
                    walk_t[i] = 0

            patches = extract_windows(video_frame, pos, window_size)
            windows = patches.astype(np.float64)

            preprocess_input(windows)
            feats = cnn.predict(windows)
            feats = feats.reshape((windows.shape[0], 512))

        for i in range(walker_count):
            if not adj[i]:

                preview_frame = np.zeros((video_shape[0], video_shape[1], 3), np.uint8)
                paint_windows(cluster_positions[i], cluster_patches[i], preview_frame, window_size)

                # Display the resulting frame 
                cv2.imshow('preview', preview_frame) 
                key = cv2.waitKey(0)
                if key == 27: # exit on ESC
                    break

                # TODO: use walker's current cluster of feats to find object in memory graph
                cluster_feats[i] = []
                cluster_positions[i] = []
                cluster_patches[i] = []

        if mask_ret == False or video_ret == False:
            break
        else:
            for i in range(walker_count):
                cluster_feats[i].append(feats[i])
                cluster_positions[i].append(pos[i])
                cluster_patches[i].append(patches[i])



run("015_chain.mp4") 