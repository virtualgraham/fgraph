######################################################################
# Using L2_Net descriptors
# One frame at a time, no batches.


from L2_Net import L2Net

import cv2
import math
import random
import math 
from memory_graph import MemoryGraph
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

def move(pos):
    m = random.randint(0, 3)

    # up
    if m == 0: 
        if pos[0] == 0:
            return move(pos)
        return (pos[0]-1, pos[1])
    # right
    elif m == 1: 
        if pos[1] == steps[1]-1:
            return move(pos)
        return (pos[0], pos[1]+1)  
    # down
    elif m == 2: 
        if pos[0] == steps[0]-1:
            return move(pos)
        return (pos[0]+1, pos[1])  
    # left
    else: 
        if pos[1] == 0:
            return move(pos)
        return (pos[0], pos[1]-1)


def resize_frame(image):
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.reshape(frame, (frame.shape[0], frame.shape[1], 1))


def filter_key_points(key_points, pos, min, n):
    dists = [math.sqrt( (kp[0] - pos[0])**2 + (kp[1]-pos[1])**2 ) for kp in key_points]
    return [x for y, x in sorted(zip(dists, key_points)) if y >= min][:n]


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

    adjacency_broken = False

    #while True: 
    rad_grid = get_rad_grid(pos, 1, shape)

    try: 
        loc = rad_grid[random.randint(0, len(rad_grid)-1)]
    except ValueError:
        print("ValueError len(rad_grid), pos, 1, shape", len(rad_grid), pos, 1, shape)
        raise

    # print(pos, loc)
    points = list(filter(lambda p: is_in_rad_grid_loc(p, loc), key_points))
    # print("len(points)", len(points))
    if len(points) > 0:
        return points[random.randint(0, len(points)-1)], adjacency_broken

    adjacency_broken = True
    pos = (loc[0] + stride/2.0, loc[1] + stride/2.0)
    return pos, adjacency_broken
            

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


def random_keypoints(frame):
    stride_blocks = (frame.shape[0] * frame.shape[1]) / float(stride * stride)
    n = int(round(random_keypoints_per_stride_block * stride_blocks))
    return [(random.random() * frame.shape[0], random.random() * frame.shape[1]) for _ in range(n)]



window_size = 64
stride=32

random_keypoints_per_stride_block = 0.05

runs = 20
max_frames=2500



def play_annotated_video():

    # Video
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture('./media/cows.mp4')

    # CNN
    l2_net = L2Net("L2Net-HP+", True)

    # initialize SIFT
    sift = cv2.xfeatures2d.SIFT_create()

    memory_graph = MemoryGraph(index_path="./data/index.bin")

    group_id = 0
    group_id_dict = {}

    while vc.isOpened():

        rval, frame = vc.read()
        if rval == False:
            break
                
        frame = resize_frame(frame)

        key_points = [(kp.pt[1],kp.pt[0]) for kp in sift.detect(frame,None)]
        key_points.extend(random_keypoints(frame))

        windows = np.empty((len(key_points), window_size, window_size, 1))

        for i in range(len(key_points)):
            windows[i] = extract_window(frame, key_points[i])

        # extract cnn features from windows
        feats = l2_net.calc_descriptors(windows)

        ids, _ = memory_graph.knn_query(feats, k = 1)

        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        for i in range(len(key_points)):
            observation_id = ids[i][0]

            # get category of observation with observation_id
            g = memory_graph.get_observation_group(observation_id)
            if g not in group_id_dict:
                group_id_dict[g] = group_id
                group_id += 1
            g = group_id_dict[g]

            print("observation_group", g)

            c = colors[g % len(colors)]

            cv2.circle(frame, (int(round(key_points[i][1])), int(round(key_points[i][0]))), 3, c, cv2.FILLED)

        cv2.imshow("preview", frame)

        key = cv2.waitKey(0)

        if key == 27: # exit on ESC
            break

    vc.release()
    cv2.destroyWindow("preview")   




def build_graph():

    print("Starting...")

    # initialize SIFT
    sift = cv2.xfeatures2d.SIFT_create()
    
    #initialize L2Net
    l2_net = L2Net("L2Net-HP+", True)

    memory_graph = MemoryGraph()

    total_frame_count = 0
    
    # for each run though the video
    for r in range(runs):

        print("Run", r)

        # open video file for a run though
        cap = cv2.VideoCapture('./media/cows.mp4')
    
        # select a random starting position
        pos = None

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


            if pos is None:
                pos = (frame.shape[0] * random.random(), frame.shape[1] * random.random())
                

            key_points = [(kp.pt[1],kp.pt[0]) for kp in sift.detect(frame,None)]
            key_points.extend(random_keypoints(frame))

            pos, adjacency_broken = next_pos(key_points, pos, frame.shape)

            window = extract_window(frame, pos)

            windows = window[np.newaxis, ...]

            # extract cnn features from windows
            feats = l2_net.calc_descriptors(windows)

            id = memory_graph.add_observation(t, pos, feats, adjacency_broken)
            cv2.imwrite('./output/testing'+str(id)+'.jpg',window)

            print("frame", t, id, pos, adjacency_broken)

            total_frame_count+=1

        cap.release()
        cv2.destroyAllWindows()

    memory_graph.save_index("./data/index.bin")
    memory_graph.close()
    
    print("Done")


build_graph()
#play_annotated_video()
