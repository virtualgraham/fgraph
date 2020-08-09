######################################################################
# Using L2_Net descriptors
# One frame at a time, no batches.


from L2_Net import L2Net

import cv2
import math
import random
import math 
from memory_graph_b import MemoryGraph, build_node_embeddings_index, db_all_edges
import numpy as np
import networkx as nx

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

    rad_grid = get_rad_grid(pos, 1, shape)

    candidates = []

    for loc in rad_grid:
        points = list(filter(lambda p: is_in_rad_grid_loc(p, loc), key_points))
        if len(points) == 0:
            points = [(loc[0] + random.random() * stride, loc[1] + random.random() * stride) for i in range(random_keypoints_per_stride_block)]
        candidates.extend(points)
    
    return candidates[random.randint(0, len(candidates)-1)]
            

def extract_windows(frame, pos):
    windows = np.empty((walker_count, window_size, window_size, 1))

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

def random_keypoints(frame):
    stride_blocks = (frame.shape[0] * frame.shape[1]) / float(stride * stride)
    n = int(round(random_keypoints_per_stride_block * stride_blocks))
    return [(random.random() * frame.shape[0], random.random() * frame.shape[1]) for _ in range(n)]

def networkx_graph():

    edges = db_all_edges()
    edge_tuples = [(e.nodes[0].id, e.nodes[1].id) for e in edges]
    print("Edges in graph", len(edge_tuples))

    G = nx.Graph()
    G.add_edges_from(edge_tuples)
    
    return G
    # n = random.choice(list(G.nodes()))

    # counts, nodes = random_walk(G, n, 10, 100)
    # print(n, counts[:10], nodes[:10])


def random_walk(graph, start, len, trials):
    visited = dict()

    for _ in range(trials):
        cur = start
        for _ in range(len):
            cur = random.choice(list(graph.neighbors(cur)))
            if cur in visited:
                visited[cur] += 1
            else:
                visited[cur] = 1
 
    nodes = []
    count = []

    for key, value in visited.items():
        nodes.append(key)
        count.append(value)

    return zip(*sorted(zip(count, nodes), reverse=True))



window_size = 64
stride=32

random_keypoints_per_stride_block = 1

runs = 1
max_frames=2500

walker_count = 10



    

    
def play_annotated_video():

    # Video
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture('./media/cows.mp4')

    # CNN
    l2_net = L2Net("L2Net-HP+", True)

    # initialize SIFT
    sift = cv2.xfeatures2d.SIFT_create()

    memory_graph = MemoryGraph(index_path="./data/index.bin")

    nx = networkx_graph()

    # group_id = 0
    # group_id_dict = {}

    while vc.isOpened():

        rval, frame = vc.read()
        if rval == False:
            break
                
        frame = resize_frame(frame)

        key_points = [(kp.pt[1],kp.pt[0]) for kp in sift.detect(frame,None)]
        random.shuffle(key_points)
        
        windows = np.empty((len(key_points), window_size, window_size, 1))

        for i in range(len(key_points)):
            windows[i] = extract_window(frame, key_points[i])

        print("windows.shape", windows.shape)

        # extract cnn features from windows
        feats = l2_net.calc_descriptors(windows)
        print("feats.shape", windows.shape)

        ids, distances = memory_graph.knn_query(feats, k = 1)

        observation_id = None

        print(distances.shape)

        for i in range(distances.shape[0]):
            # print(distances[i][0])
            if distances[i][0] < 0.95:
                random_keypoint = key_points[i]
                observation_id = ids[i][0]
                print("distances[i][0]", distances[i][0])
                break
       
        if observation_id is None:
           print("no close observation found")
           continue

        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        counts, node_ids = random_walk(nx, observation_id, 100, 1000)

        n = 0
        for i in range(len(counts)):
            count = counts[i]
            if count < 200:
                break
            n += 1

        nodes = memory_graph.get_nodes(list(node_ids[:n]))

        for i in range(n):
            node = nodes[i]
            x = node["x"]
            y = node["y"]
            t = node["t"]
            cv2.circle(frame, (int(round(x)), int(round(y))), 3, colors[0], cv2.FILLED)

        cv2.circle(frame, (int(round(random_keypoint[1])), int(round(random_keypoint[0]))), 7, colors[2], cv2.FILLED)

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
            feats = l2_net.calc_descriptors(windows)

            ids = memory_graph.add_parrelell_observations(t, pos, feats)

            for i in range(walker_count):
                cv2.imwrite('./output/testing'+str(ids[i])+'.jpg',windows[i])
                
            total_frame_count+=1

        cap.release()
        cv2.destroyAllWindows()

    memory_graph.save_index("./data/index.bin")
    memory_graph.close()
    
    print("Done")



        

        

#build_graph()
play_annotated_video()
#build_node_embeddings_index()
#random_walk_test()