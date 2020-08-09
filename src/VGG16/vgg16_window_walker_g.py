######################################################################


from keras.applications import vgg16
from keras.applications.vgg16 import preprocess_input

import cv2
import math
import random
import math 
from memory_graph_f import MemoryGraph, MemoryGraphWalker
import numpy as np


# randomly walk window over frames of video
# calculate CNN features for each window

colors = [
    (1, 0, 103),
    (213, 255, 0),
    (255, 0, 86),
    (158, 0, 142),
    (14, 76, 161),
    (255, 229, 2),
    (0, 95, 57),
    (0, 255, 0),
    (149, 0, 58),
    (255, 147, 126),
    (164, 36, 0),
    (0, 21, 68),
    (145, 208, 203),
    (98, 14, 0),
    (107, 104, 130),
    (0, 0, 255),
    (0, 125, 181),
    (106, 130, 108),
    (0, 174, 126),
    (194, 140, 159),
    (190, 153, 112),
    (0, 143, 156),
    (95, 173, 78),
    (255, 0, 0),
    (255, 0, 246),
    (255, 2, 157),
    (104, 61, 59),
    (255, 116, 163),
    (150, 138, 232),
    (152, 255, 82),
    (167, 87, 64),
    (1, 255, 254),
    (255, 238, 232),
    (254, 137, 0),
    (189, 198, 255),
    (1, 208, 255),
    (187, 136, 0),
    (117, 68, 177),
    (165, 255, 210),
    (255, 166, 254),
    (119, 77, 0),
    (122, 71, 130),
    (38, 52, 0),
    (0, 71, 84),
    (67, 0, 44),
    (181, 0, 255),
    (255, 177, 103),
    (255, 219, 102),
    (144, 251, 146),
    (126, 45, 210),
    (189, 211, 147),
    (229, 111, 254),
    (222, 255, 116),
    (0, 255, 120),
    (0, 155, 255),
    (0, 100, 1),
    (0, 118, 255),
    (133, 169, 0),
    (0, 185, 23),
    (120, 130, 49),
    (0, 255, 198),
    (255, 110, 65),
    (232, 94, 190),
    (0, 0, 0),
]




def resize_frame(image):
    #frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #return np.reshape(frame, (frame.shape[0], frame.shape[1], 1))
    return image

def is_pos_in_frame(pos, shape):
    return (pos[0] + window_size/2.0 >= 0 and pos[1] + window_size/2.0 >= 0 and pos[0] - window_size/2.0 < shape[0] and pos[1] - window_size/2.0 < shape[1]) 


def get_rad_grid(pos, rad, shape):

    top_left = (pos[0] - (rad+.5)*stride, pos[1] - (rad+.5)*stride)

    res = []

    for i in range(2*rad+1):
        p = (top_left[0]+i*stride, top_left[1])
        if is_pos_in_frame(p, shape):
            res.append(p)
 
    for i in range(2*rad+1):
        p = (top_left[0]+i*stride, top_left[1]+(2*rad+1)*stride)
        if is_pos_in_frame(p, shape):
            res.append(p)

    for i in range(2*rad-1):
        p = (top_left[0], top_left[1]+(i+1)*stride)
        if is_pos_in_frame(p, shape):
            res.append(p)

    for i in range(2*rad-1):
        p = (top_left[0]+(2*rad)*stride, top_left[1]+(i+1)*stride)
        if is_pos_in_frame(p, shape):
            res.append(p)

    return res


def is_in_rad_grid_loc(pos, rad_grid_loc):
    # print(pos, rad_grid_loc)
    return (pos[0] >= rad_grid_loc[0] and 
        pos[0] < (rad_grid_loc[0] + stride) and
        pos[1] >= rad_grid_loc[1] and 
        pos[1] < (rad_grid_loc[1] + stride))


def first_pos(shape):
    return (random.random() * shape[0], random.random() * shape[1])


def next_pos(pos, shape):
    if pos is None:
        return first_pos(shape)

    rad_grid = get_rad_grid(pos, 1, shape)
    g = random.choice(rad_grid)
    return (g[0] + random.random() * stride, g[1] + random.random() * stride)


def extract_windows(frame, pos):
    windows = np.empty((walker_count, window_size, window_size, 3))

    for i in range(walker_count):
        windows[i] = extract_window(frame, pos[i])

    return windows


def extract_window(frame, pos):
    half_w = window_size/2.0
    bottom_left = [int(round(pos[0]-half_w)), int(round(pos[1]-half_w))]
    top_right = [bottom_left[0]+window_size, bottom_left[1]+window_size]
    if bottom_left[0] < 0:
        #print("bottom_left[0] < 0")
        top_right[0] -= bottom_left[0]
        bottom_left[0] = 0
    if bottom_left[1] < 0:
        #print("bottom_left[1] < 0")
        top_right[1] -= bottom_left[1]
        bottom_left[1] = 0
    if top_right[0] >= frame.shape[0]:
        #print("top_right[0] >= frame.shape[0]")
        bottom_left[0] -= (top_right[0]-frame.shape[0]+1)
        top_right[0] = frame.shape[0]-1
    if top_right[1] >= frame.shape[1]:
        #print("top_right[1] >= frame.shape[1]")
        bottom_left[1] -= (top_right[1]-frame.shape[1]+1)
        top_right[1] = frame.shape[1]-1
        

    # print(pos, bottom_left, top_right)
    return frame[bottom_left[0]:top_right[0], bottom_left[1]:top_right[1]]







def paint_windows(positions, windows, frame, rect=-1):
    for i in range(len(positions)):
        pos = positions[i]
        x1 = int(round(pos[1] - window_size/2.0))
        x2 = x1 + window_size
        y1 = int(round(pos[0] - window_size/2.0))
        y2 = y1 + window_size
        
        window = windows[i]

        # print(abs(y1-y2) != window_size and abs(x1-x2) != window_size, abs(y1-y2), abs(x1-x2))

        if abs(y1-y2) != window_size and abs(x1-x2) != window_size:
            continue

        wx1 = 0
        wx2 = window_size
        wy1 = 0
        wy2 = window_size

        shape = frame.shape

        if y1 < 0:
            if y1 < -window_size:
                continue
            wy1 = -y1
            y1 = 0
        if y2 >= shape[0]:
            if y2 >= (shape[0] + window_size):
                continue
            wy2 = window_size - (y2 - shape[0] + 1)
            y2 = shape[0]-1 
        if x1 < 0:
            if x1 < -window_size:
                continue
            wx1 = -x1
            x1 = 0
        if x2 >= shape[1]:
            if x2 >= (shape[1] + window_size):
                continue
            wx2 = window_size - (x2 - shape[1] + 1)
            x2 = shape[1]-1

        frame[y1:y2, x1:x2] = window[wy1:wy2, wx1:wx2]

        if rect > -1:
            x1 = int(round(pos[1] - window_size/2.0))
            x2 = int(round(pos[0] - window_size/2.0))
            y1 = x1 + window_size
            y2 = x2 + window_size

            cv2.rectangle(frame, (y1, y2), (x1,x2), colors[rect % len(colors)], 1)


def show_patches(path_windows, path_features, path_positions, frame_shape, memory_graph):
    print("show_patches")

    cv2.namedWindow("patches")

    frame = np.zeros((frame_shape[0], frame_shape[1], 3), np.uint8)

    paint_windows(path_positions, path_windows, frame, 0)

    # features, cos_dis, depth, min_matches
    groups = list(memory_graph.search_group(path_features, 0.2, 2, 5))

    print("groups", groups)

    for i in range(len(groups)):
        group = list(groups[i])
        windows = np.array([cv2.imread('./patches/patch'+str(nid)+'.png') for nid in group])

        observations = memory_graph.get_observations(group)
        positions = [(obs["y"], obs["x"]) for obs in observations]

        paint_windows(positions, windows, frame, i+1)

    cv2.imshow('patches', frame) 



def play_video():

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONUP:
            return

        res_frame = resize_frame(frame)

        pos = (y, x)

        path = [pos]

        for i in range(playback_random_walk_length):
            path.append(next_pos(pos, frame.shape))

        path = list(set(path))

        windows = np.array([extract_window(res_frame, p) for p in path])

        preprocess_input(windows)
        features = model.predict(windows)
        features = features.reshape((windows.shape[0], 512))

        print("windows.shape, feats.shape", windows.shape, features.shape)
        show_patches(windows, features, path, frame.shape, memory_graph)

    memory_graph = MemoryGraph(graph_path=graph_file, index_path=index_file, space='cosine', dim=512)

    model = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(32, 32, 3))

    cap = cv2.VideoCapture(video_file) 
   
    # Check if camera opened successfully 
    if (cap.isOpened() == False):  
        print("Error opening video  file") 

    cv2.namedWindow("preview")
    cv2.setMouseCallback("preview", on_click)

    # Read until video is completed 
    while(cap.isOpened()): 
        
        # Capture frame-by-frame 
        ret, frame = cap.read() 
        
        if ret == True: 
            
            # Display the resulting frame 
            cv2.imshow('preview', frame) 
        
            # Press Q on keyboard to  exit 
            key = cv2.waitKey(0)

            if key == 27: # exit on ESC
                break

        # Break the loop 
        else:  
            break
    
    # When everything done, release  
    # the video capture object 
    cap.release() 
    
    # Closes all the frames 
    cv2.destroyAllWindows() 





def build_graph():

    print("Starting...")

    # initialize VGG16
    model = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(32, 32, 3))

    # memory graph
    memory_graph = MemoryGraph(space='cosine', dim=512)
    memory_graph_walker = MemoryGraphWalker(memory_graph, distance_threshold = 0.10,  identical_distance = 0.01)

    total_frame_count = 0
    
    # for each run though the video
    for r in range(runs):

        print("Run", r)

        # open video file for a run though
        cap = cv2.VideoCapture(video_file)
    
        # walkers
        pos = [None for _ in range(walker_count)]
        
        done = False

        # for each frame
        for t in range(max_frames):
            if done:
                break

            ret, frame = cap.read()
                
            if ret == False:
                done = True
                break

            frame = resize_frame(frame)

            for i in range(walker_count):
                pos[i] = next_pos(pos[i], frame.shape)

            windows = extract_windows(frame, pos)

            # extract cnn features from windows
            preprocess_input(windows)
            feats = model.predict(windows)
            feats = feats.reshape((windows.shape[0], 512))

            print("feats.shape", feats.shape)

            ids = memory_graph_walker.add_parrelell_observations(t, pos, feats)

            if save_windows:
                for i in range(walker_count):
                    cv2.imwrite('./patches/patch'+str(ids[i])+'.png',windows[i])
                
            total_frame_count+=1

        cap.release()
        cv2.destroyAllWindows()

    memory_graph.save_graph(graph_file)
    memory_graph.save_index(index_file)
    
    print("Done")


playback_random_walk_length = 10

window_size = 32
stride=16
random_keypoints_per_stride_block = 1
runs = 1
max_frames=30*15
walker_count = 100

video_file = './media/l8jJQA1h.mp4'
graph_file = "./data/tt_graph.pickle"
index_file = "./data/tt_index.bin"

save_windows = True




build_graph()
#play_video()