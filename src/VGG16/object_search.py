import numpy as np
import cv2
from os.path import join
import random

from vgg16_window_walker_d import color_fun, extract_window, get_rad_grid


max_pos_attempts = 1000
mask_path = "./media/tabletop_objects/masks/"
video_path = "./media/tabletop_objects/videos/"
db_path = "./data/table_objects_10.db"
max_frames = 30*30
walker_count = 3
window_size = 32
stride = 16
center_size = 16
center_threshold = center_size*center_size*0.9
grid_margin = 16
walk_length = 10


def key_point_grid(orb, frame, obj_frame, stride):

    grid_width = Math.floor((frame.shape[0] - grid_margin*2) / stride)
    grid_height = Math.floor((frame.shape[1] - grid_margin*2) / stride)

    grid_offset_x = ((frame.shape[0] - grid_margin*2) % stride)/2.0 + grid_margin
    grid_offset_y = ((frame.shape[1] - grid_margin*2) % stride)/2.0 + grid_margin

    object_grid_locations = set()

    for x in range(grid_width):
        for y in range(grid_height):
            p = (grid_offset_x + x * stride + 0.5 * stride, grid_offset_y + y * stride + 0.5 * stride)
            w = extract_window(obj_frame, p, stride)
            if extract_object(w, stride) is not None:
                object_grid_locations.add((x, y))
            
    kp = orb.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)

    grid = dict()

    for k in kp:
        g = (int(math.floor((p[0]-grid_offset_x)/stride)), int(math.floor((p[1]-grid_offset_y)/stride)))
        if g in object_grid_locations:
            p = (k.pt[1],k.pt[0])
            if g in grid:
                grid[g].append(p)
            else:
                grid[g] = [p]

    return grid


def first_pos(kp_grid):
    loc = random.choice(list(kp_grid.keys()))
    return loc, random.choice(kp_grid[loc])


def next_pos(kp_grid, shape, g_pos, walk_length, stride):
 
    if (g_pos is not None) and (random.random() > 1.0/walk_length):

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

    memory_graph = MemoryGraph(db_path, space='cosine', dim=512, max_elements=max_elements)
    cnn = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(32, 32, 3))
    orb = cv2.ORB_create(nfeatures=100000, fastThreshold=7)

    mask = cv2.VideoCapture(join(mask_path,"mask_"+file))
    video = cv2.VideoCapture(join(video_path,file))

    g_pos = [None for _ in range(walker_count)]
    pos = [None for _ in range(walker_count)]
    adj = [False for _ in range(walker_count)]
    feat_clusters = [[] for _ in range(walker_count)]

    for t in range(max_frames):
        mask_ret, mask_frame = mask.read()
        video_ret, video_frame = video.read()

        if mask_ret == False or video_ret == False:
            adj = [False for _ in range(walker_count)]
        else:
            obj_frame = color_fun(mask_frame)
            kp_grid = key_point_grid(orb, video_frame, stride)

            for i in range(walker_count):
                g_pos[i], pos[i], adj[i] = next_pos(obj_frame, kp_grid, g_pos[i], walk_length, stride)

            patches = extract_windows(video_frame, pos, window_size)
            windows = patches.astype(np.float64)

            preprocess_input(windows)
            feats = cnn.predict(windows)
            feats = feats.reshape((windows.shape[0], 512))

        for i in range(walker_count):
            if not adj[i]:
                # TODO: use walker's current cluster of feats to find object in memory graph
                feat_clusters[i] = []

        if mask_ret == False or video_ret == False:
            break
        else:
            feat_clusters[i].append(feats[i])



run("015_chain.mp4") 