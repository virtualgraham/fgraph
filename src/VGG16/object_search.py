import numpy as np
import cv2
from os.path import join
import random
import math 
import re 
from datetime import datetime

from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.vgg16 import preprocess_input

from vgg16_window_walker_lib_h import color_fun, extract_windows, extract_window, extract_object, get_rad_grid, paint_windows, MemoryGraph, extract_window_pixels

from itertools import chain

mask_path = "../../media/tabletop_objects/masks/"
video_path = "../../media/tabletop_objects/videos/"
db_path = "../../data/table_objects_h.db"
max_frames = 30
walker_count = 4
window_size = 32
stride = 16
center_size = 16
center_threshold = center_size*center_size*0.9
grid_margin = 16
walk_length = 10
max_elements=1200000

def key_point_grid(orb, frame, obj_frame, stride):

    grid_width = math.floor((frame.shape[0] - grid_margin*2) / stride)
    grid_height = math.floor((frame.shape[1] - grid_margin*2) / stride)

    grid_offset_x = ((frame.shape[0] - grid_margin*2) % stride)/2.0 + grid_margin
    grid_offset_y = ((frame.shape[1] - grid_margin*2) % stride)/2.0 + grid_margin

    object_grid_locations = set()

    #print("grid_width", grid_width, "grid_height", grid_height)
    for x in range(grid_width):
        for y in range(grid_height):
            p = (grid_offset_x + x * stride + 0.5 * stride, grid_offset_y + y * stride + 0.5 * stride)
            w = extract_window(obj_frame, p, stride)
            if extract_object(w, stride) is not None:
                object_grid_locations.add((x, y))
    
    #print("len(object_grid_locations)", len(object_grid_locations))
    kp = orb.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
    #print("len(kp)", len(kp))

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
    
    if (g_pos is not None) and walk_t < (walk_length-1):

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


def search(files):
    memory_graph = MemoryGraph(db_path, space='cosine', dim=512, max_elements=max_elements)
    cnn = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(32, 32, 3))
    orb = cv2.ORB_create(nfeatures=100000, fastThreshold=7)

    for file in files:
        search_file(file, memory_graph, cnn, orb)


def search_file(file, memory_graph, cnn, orb):
    print("search", file)

    mask = cv2.VideoCapture(join(mask_path,"mask_"+file))
    video = cv2.VideoCapture(join(video_path,file))

    g_pos = [None for _ in range(walker_count)]
    pos = [None for _ in range(walker_count)]
    adj = [False for _ in range(walker_count)]
    walk_t = [0 for _ in range(walker_count)]
    cluster_feats = [[] for _ in range(walker_count)]
    cluster_positions = [[] for _ in range(walker_count)]
    cluster_patches = [[] for _ in range(walker_count)]

    observation_ids = set()

    done = False

    for t in range(max_frames):
        if t % walk_length == 0:
            print("frame", t)

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
        
        if mask_ret == False or video_ret == False:
            done = True
        else:
            for i in range(walker_count):
                cluster_feats[i].append(feats[i])
                cluster_positions[i].append(pos[i])
                cluster_patches[i].append(patches[i])

        for i in range(walker_count):
            if (not adj[i] or done) and len(cluster_feats[i]) > 0:

                ########
                
                similar_clusters = memory_graph.search_group(cluster_feats[i], k=30)
                node_ids = set(chain.from_iterable(similar_clusters))
                observation_ids.update(memory_graph.observations_for_nodes(node_ids))
                
                ########

                cluster_feats[i] = []
                cluster_positions[i] = []
                cluster_patches[i] = []
        
        if done:
            break



    observations = memory_graph.get_observations(observation_ids)

    object_name = re.findall('_([a-z]+)', file)[0]


    labeled_observations = 0
    mislabeled_observations = 0

    labeled_video_files = set()
    mislabeled_video_files = set()

    video_files = dict()

    for o in observations:
        if "o" in o and o["o"] == object_name:
            labeled_observations += 1
        else:
            mislabeled_observations += 1

        if object_name in o["file"]:
            labeled_video_files.add(o["file"])
        else:
            mislabeled_video_files.add(o["file"])
        
        if o["file"] not in video_files:
            video_files[o["file"]] = dict()
        
        video_file = video_files[o["file"]]

        if o["t"] not in video_file:
            video_file[o["t"]] = {
                True: set(),
                False: set(),
            }

        if "o" in o and o["o"] == object_name:
            video_frame = video_file[o["t"]][True]
        else:
            video_frame = video_file[o["t"]][False]

        video_frame.update(extract_window_pixels((o["y"], o["x"]), video_shape, window_size))
        

    labeled_frames = 0
    mislabeled_frames = 0

    labeled_pixels = 0
    mislabeled_pixels = 0

    # TODO: what if it returns an observation on a frame in the wrong location but that contains the object elsewhere?
    for video_file in video_files.values():
        for video_frame in video_file.values():
            if len(video_frame[True]) > 0:
                labeled_frames += 1
            else:
                mislabeled_frames += 1

            labeled_pixels += len(video_frame[True])
            # TODO: should intersections be taken from mislabeled?
            mislabeled_pixels += len(video_frame[False] - video_frame[True])

    
    print("")
    print(datetime.now())
    print(object_name)
    print("len(observations)", len(observations))
    print("")

    counts = memory_graph.get_counts()


    if len(observations) > 0:
        observations_of_sample_class = labeled_observations / len(observations)
        observations_not_of_sample_class = 1 - observations_of_sample_class
    else:
        observations_of_sample_class = None
        observations_not_of_sample_class = None
    
    print("observations_of_sample_class", observations_of_sample_class, labeled_observations, len(observations))
    print("observations_not_of_sample_class", observations_not_of_sample_class)
    
    print("")

    if object_name in counts["observation_objects"] and counts["observation_objects"][object_name] > 0:
        observations_true_positive = labeled_observations / counts["observation_objects"][object_name]
        observations_false_negative = 1 - observations_true_positive
    else:
        observations_true_positive = None
        observations_false_negative = None

    print("observations_true_positive", observations_true_positive, labeled_observations, counts["observation_objects"][object_name] if object_name in counts["observation_objects"] else None)
    print("observations_false_negative", observations_false_negative)

    if object_name in counts["observation_objects"] and (counts["observation_count"] - counts["observation_objects"][object_name]) > 0:
        observations_false_positive = mislabeled_observations / (counts["observation_count"] - counts["observation_objects"][object_name])
        observations_true_negative = 1 - observations_false_positive
    else:
        observations_false_positive = None
        observations_true_negative = None

    print("observations_false_positive", observations_false_positive, mislabeled_observations, (counts["observation_count"] - counts["observation_objects"][object_name]) if object_name in counts["observation_objects"] else None)
    print("observations_true_negative", observations_true_negative)
   
    print("")

    if object_name in counts["video_objects"] and counts["video_objects"][object_name] > 0:
        video_true_positive = len(labeled_video_files) / counts["video_objects"][object_name]
        video_false_negative = 1 - video_true_positive
    else:
        video_true_positive = None
        video_false_negative = None

    print("video_true_positive", video_true_positive, len(labeled_video_files), counts["video_objects"][object_name] if object_name in counts["video_objects"] else None)
    print("video_false_negative", video_false_negative)
    
    if object_name in counts["video_objects"] and (counts["video_count"] - counts["video_objects"][object_name]) > 0:
        video_false_positive = len(mislabeled_video_files) / (counts["video_count"] - counts["video_objects"][object_name])
        video_true_negative = 1 - video_false_positive
    else:
        video_false_positive = None
        video_true_negative = None

    print("video_false_positive", video_false_positive, len(mislabeled_video_files), (counts["video_count"] - counts["video_objects"][object_name]) if object_name in counts["video_objects"] else None)
    print("video_true_negative", video_true_negative)
    
    print("")

    if object_name in counts["frame_objects"] and counts["frame_objects"][object_name] > 0:
        frame_true_positive = labeled_frames / counts["frame_objects"][object_name]
        frame_false_negative = 1 - frame_true_positive
    else:
        frame_true_positive = None
        frame_false_negative = None

    print("frame_true_positive", frame_true_positive, labeled_frames, counts["frame_objects"][object_name] if object_name in counts["frame_objects"] else None)
    print("frame_false_negative", frame_false_negative)
    
    if object_name in counts["frame_objects"] and (counts["frame_count"] - counts["frame_objects"][object_name]) > 0:
        frame_false_positive = mislabeled_frames / (counts["frame_count"] - counts["frame_objects"][object_name])
        frame_true_negative = 1 - frame_false_positive
    else:
        frame_false_positive = None
        frame_true_negative = None

    print("frame_false_positive", frame_false_positive, mislabeled_frames, (counts["frame_count"] - counts["frame_objects"][object_name]) if object_name in counts["frame_objects"] else None)
    print("frame_true_negative", frame_true_negative)
    
    print("")

    if object_name in counts["pixel_objects"] and counts["pixel_objects"][object_name] > 0:
        pixel_true_positive = labeled_pixels / counts["pixel_objects"][object_name]
        pixel_false_negative = 1 - pixel_true_positive
    else:
        pixel_true_positive = None
        pixel_false_negative = None

    print("pixel_true_positive", pixel_true_positive, labeled_pixels, counts["pixel_objects"][object_name] if object_name in counts["pixel_objects"] else None)
    print("pixel_false_negative", pixel_false_negative)
    

    if object_name in counts["pixel_objects"] and (counts["pixel_count"] - counts["pixel_objects"][object_name]) > 0:
        pixel_false_positive = mislabeled_pixels / (counts["pixel_count"] - counts["pixel_objects"][object_name])
        pixel_true_negative = 1 - pixel_false_positive
    else:
        pixel_false_positive = None
        pixel_true_negative = None

    print("pixel_false_positive", pixel_false_positive, mislabeled_pixels, (counts["pixel_count"] - counts["pixel_objects"][object_name]) if object_name in counts["pixel_objects"] else None)
    print("pixel_true_negative", pixel_true_negative)
    
    print("")



files = [
    '001_apple.mp4',
    '002_apple.mp4',
    '003_apple.mp4',
    '004_bear.mp4',
    '005_bear.mp4',
    '006_bear.mp4',
    '007_brush.mp4',
    '008_brush.mp4',
    '009_brush.mp4',
    '010_carrot.mp4',
    '011_carrot.mp4',
    '012_carrot.mp4',
    '013_chain.mp4',
    '014_chain.mp4',
    '015_chain.mp4',
    '016_clippers.mp4',
    '017_clippers.mp4',
    '018_clippers.mp4',
    '019_cologne.mp4',
    '020_cologne.mp4',
    '021_cologne.mp4',
    '022_cup.mp4',
    '023_cup.mp4',
    '024_cup.mp4',
    '025_flowers.mp4',
    '026_flowers.mp4',
    '027_flowers.mp4',
    '028_hanger.mp4',
    '029_hanger.mp4',
    '030_hanger.mp4',
    '031_ketchup.mp4',
    '032_ketchup.mp4',
    '033_ketchup.mp4',
    '034_notebook.mp4',
    '035_notebook.mp4',
    '036_notebook.mp4',
    '037_opener.mp4',
    '038_opener.mp4',
    '039_opener.mp4',
    '040_pepper.mp4',
    '041_pepper.mp4',
    '042_pepper.mp4',
    '043_rock.mp4',
    '044_rock.mp4',
    '045_rock.mp4',
    '046_shorts.mp4',
    '047_shorts.mp4',
    '048_shorts.mp4',
]


search(files)