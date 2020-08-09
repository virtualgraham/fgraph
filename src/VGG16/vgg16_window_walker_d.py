from keras.applications import vgg16
from keras.applications.vgg16 import preprocess_input

import cv2
import math
import random
import math 
from memory_graph_d import MemoryGraph, MemoryGraphWalker
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
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def is_pos_in_shape(pos, shape):
    return pos[0] >= 0 and pos[1] >= 0 and pos[0] <= shape[0] and pos[1] <= shape[1]

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


def next_pos(key_points, pos, shape):
    if(len(key_points) == 0):
        return pos

    rad_grid = get_rad_grid(pos, 1, shape)

    candidates = []

    for loc in rad_grid:
        points = list(filter(lambda p: is_in_rad_grid_loc(p, loc), key_points))
        if len(points) == 0:
            points = [(loc[0] + random.random() * stride, loc[1] + random.random() * stride) for i in range(random_keypoints_per_stride_block)]
            points = list(filter(lambda p: is_pos_in_shape(p, shape), key_points))
        candidates.extend(points)
    
    return candidates[random.randint(0, len(candidates)-1)]
            

def extract_windows(frame, pos):
    windows = np.empty((walker_count, window_size, window_size, 3))

    for i in range(walker_count):
        windows[i] = extract_window(frame, pos[i])

    return windows


def extract_window(frame, pos):

    window = np.zeros((window_size, window_size ,3))

    wxa = 0
    wxb = window_size
    wya = 0
    wyb = window_size

    fxa = int(round(pos[0]-window_size/2.0))
    fxb = fxa + window_size
    fya = int(round(pos[1]-window_size/2.0))
    fyb = fya + window_size

    if fxa < 0:
        #print("fxa < 0")
        wxa = -fxa
        fxa = 0
        
    if fya < 0:
        #print("fya < 0")
        wya = -fya
        fya = 0

    if fxb > frame.shape[0]:
        #print("fxb > frame.shape[0]")
        wxb = wxb - (fxb - frame.shape[0])
        fxb = frame.shape[0]

    if fyb > frame.shape[1]:
        #print("fyb > frame.shape[1]")
        wyb = wyb - (fyb - frame.shape[1])
        fyb = frame.shape[1]

    window[wxa:wxb, wya:wyb] = frame[fxa:fxb, fya:fyb]

    #print(wxa,wxb, wya,wyb, fxa,fxb, fya,fyb)

    window_masked = np.zeros((window_size, window_size ,3))
    window_masked[32*2:32*5, 32*2:32*5] = window[32*2:32*5, 32*2:32*5]

    return window_masked


window_size = 224
stride=32
random_keypoints_per_stride_block = 1
runs = 1
max_frames=30*90

walker_count = 10

video_file = './media/fish.mp4'
graph_file = "./data/fish_graph.pickle"
index_file = "./data/fish_index.bin"

save_windows = True

def play_annotated_video():

    def on_click(event, x, y, flags, param):

        if event != cv2.EVENT_LBUTTONUP:
            return
            
        rval, frame = vc.read()
        if rval == False:
            return
                
        frame = resize_frame(frame)

        key_points = [(kp.pt[1],kp.pt[0]) for kp in sift.detect(frame,None)]

        loc = (y-stride/2.0, x-stride/2.0)
        key_points = list(filter(lambda p: is_in_rad_grid_loc(p, loc), key_points))
    
        if len(key_points) == 0:
            print("no keypoints there")
            return

        random.shuffle(key_points)

        windows = np.empty((len(key_points), window_size, window_size, 3))

        for i in range(len(key_points)):
            windows[i] = extract_window(frame, key_points[i])

        print("windows.shape", windows.shape)

        # extract cnn features from windows
        
        windows = preprocess_input(windows)
        feats = model.predict(windows)[:, 3, 3, :]

        ids, distances = memory_graph.knn_query(feats, k = 1)

        observation_id = None

        print("distances.shape", distances.shape)

        for i in range(distances.shape[0]):
            print("distances[i][0]", distances[i][0])
            if distances[i][0] < 0.1:
                random_keypoint = key_points[i]
                observation_id = ids[i][0]
                print("distances[i][0]", distances[i][0])
                break
       
        if observation_id is None:
           print("no close observation found")
           return

        counts, node_ids = memory_graph.random_walk(observation_id, 100, 1000)

        n = 0
        for i in range(len(counts)):
            count = counts[i]
            if count < 200:
                break
            n += 1

        nodes = memory_graph.get_observations(node_ids[:n])

        for i in range(n):
            node = nodes[i]
            x = node["x"]
            y = node["y"]
            t = node["t"]
            cv2.circle(frame, (int(round(x)), int(round(y))), 3, colors[0], cv2.FILLED)

        cv2.circle(frame, (int(round(random_keypoint[1])), int(round(random_keypoint[0]))), 7, colors[2], cv2.FILLED)

        cv2.imshow("preview", frame)

        key = cv2.waitKey(0)

        # if not vc.isOpened():
        #     vc.release()
        #     cv2.destroyWindow("preview")   

    # Video
    cv2.namedWindow("preview")
    cv2.setMouseCallback("preview", on_click)

    vc = cv2.VideoCapture(video_file)

    # CNN
    model = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

    # initialize SIFT
    sift = cv2.xfeatures2d.SIFT_create()

    memory_graph = MemoryGraph(graph_path=graph_file, index_path=index_file, space='cosine', dim=512)

    while vc.isOpened():
        rval, frame = vc.read()
        cv2.imshow("preview", frame)
        key = cv2.waitKey(0)






def build_graph():

    print("Starting...")

    # initialize SIFT
    sift = cv2.xfeatures2d.SIFT_create()
    
    # initialize VGG16
    model = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(32, 32, 3))

    print(model.summary())

    memory_graph = MemoryGraph(space='cosine', dim=512)
    memory_graph_walker = MemoryGraphWalker(memory_graph, distance_threshold = 0.15, identical_distance = 0.015)

    total_frame_count = 0
    
    # for each run though the video
    for r in range(runs):

        print("Run", r)

        # open video file for a run though
        cap = cv2.VideoCapture(video_file)
    
        # select a random starting position
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
                if pos[i] is None:
                    pos[i] = (frame.shape[0] * random.random(), frame.shape[1] * random.random())
                
            key_points = [(kp.pt[1],kp.pt[0]) for kp in sift.detect(frame,None)]

            for i in range(walker_count):
                pos[i] = next_pos(key_points, pos[i], frame.shape)

            windows = extract_windows(frame, pos)

            # extract cnn features from windows
            preprocess_input(windows)
            feats = model.predict(windows)
            print("feats.shape", feats.shape)

            ids = memory_graph_walker.add_parrelell_observations(t, pos, feats)

            if save_windows:
                for i in range(walker_count):
                    cv2.imwrite('./output/testing'+str(ids[i])+'.jpg',windows[i])
                
            total_frame_count+=1

        cap.release()
        cv2.destroyAllWindows()

    memory_graph.save_graph(graph_file)
    memory_graph.save_index(index_file)
    
    print("Done")




build_graph()
#play_annotated_video()
