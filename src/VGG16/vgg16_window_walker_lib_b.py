import math
import random
import time
import struct
import os.path
from os import listdir
from os.path import isfile, isdir, join, split, splitext
import json
from pathlib import Path

import numpy as np
from scipy import spatial
from sklearn import preprocessing
import cv2
import hnswlib
import networkx as nx
import plyvel

from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.vgg16 import preprocess_input


class MemoryGraphWalker:
    def __init__(self, memory_graph, knn = 30, accurate_prediction_limit = 10, distance_threshold = 0.1,  adjacency_radius = 2, identical_distance=0.01):

        self.knn = knn
        self.accurate_prediction_limit = accurate_prediction_limit
        self.identical_distance = identical_distance
        self.distance_threshold = distance_threshold
        self.adjacency_radius = adjacency_radius

        self.memory_graph = memory_graph
        self.last_ids = dict()
        self.predictions = dict()

    def add_parrelell_observations(self, file, t, pos, adj, feats):
        return [self.add_observation(file, t, pos[i], adj[i], feats[i], i) for i in range(len(feats))]

    def add_observation(self, file, t, pos, adj, feats, walker_id):
 
        stats = {"adj":adj}

        observation_id = self.memory_graph.insert_observation({"file":file, "t":t, "y":pos[0], "x":pos[1]})

        l = d = None

        if self.memory_graph.index_count() >= self.knn:
            labels, distances = self.memory_graph.knn_query([feats], k = self.knn)
            l = labels[0]
            d = distances[0]

        if d is not None:
            #print("Nearest Neighbor", d[0])
            stats["nearest_neighbor"] = d[0]

        accurate_predictions = set()
        evaluated_ids = set()

        # find correct predictions and reinforce with adjacency
        if adj and walker_id in self.predictions:
            predictions = self.predictions[walker_id]

            for pred in predictions:
                a = pred["id_similar_to_prev"]
                b = pred["id_similar_to_curr"]

                if b in evaluated_ids:
                    continue

                # these features come from the index and are normalized
                f = pred['candidate_for_similar_to_curr']["f"]

                # TODO: normalize feats before calling distance
                if self.memory_graph.distance(feats, f) <= self.distance_threshold:
                    # print("add_predicted_observations", b, observation_id)
                    self.memory_graph.add_predicted_observations([b], [observation_id])
                    accurate_predictions.add(a)
                
                if len(accurate_predictions) >= self.accurate_prediction_limit:
                    # print("Too many accurate_predictions")
                    break

                evaluated_ids.add(b)

            if len(predictions) > 0:
                stats["predictions"] = len(predictions)
                stats["accurate_predictions"] = len(accurate_predictions)
                # print("Predictions", len(accurate_predictions), "of", len(predictions))
        
        # print("frame", t, pos)

        if len(accurate_predictions) < self.accurate_prediction_limit:
            
            #if d is not None and sum([i < self.identical_distance for i in d]) > 3:
            if d is not None and (d[0] < self.identical_distance):
                node_id = l[0]
                stats["identical"] = True
                #print("Using Identical")
            else:
                node_id = self.memory_graph.insert_node({"f":feats})
            
            #print("NodeID", node_id)

            self.memory_graph.add_integrated_observations([node_id], [observation_id])
    
            stats["adjacencies_inserted"] = 0
            if adj:
                if walker_id in self.last_ids and self.last_ids[walker_id] is not None :
                    stats["adjacencies_inserted"] += 1
                    self.memory_graph.insert_adjacency(self.last_ids[walker_id], node_id)

                for a in accurate_predictions:
                    stats["adjacencies_inserted"] += 1
                    self.memory_graph.insert_adjacency(a, node_id)
            
        else:
            node_id = None

        # make predictions
        self.predictions[walker_id] = []

        if self.memory_graph.index_count() >= self.knn and l is not None and d is not None:

            similar = 0
            
            for n in range(self.knn):
                label = l[n]
                distance = d[n]
                
                if distance <= self.distance_threshold:
                    # Found a previous similar observation

                    # find other observations that have been seen near this one
                    next_adjacencies = self.memory_graph.get_adjacencies(label, self.adjacency_radius)

                    for n in next_adjacencies:
                        props = self.memory_graph.get_node(n)
                        self.predictions[walker_id].append(dict(id_similar_to_prev=label, id_similar_to_curr=n, candidate_for_similar_to_curr=props))

                    similar += 1


        self.last_ids[walker_id] = node_id

        return node_id, observation_id, stats
        


class MemoryGraph:

    def __init__(self, path, space='cosine', dim=512, max_elements=1000000, ef=100, M=48):
        self.space = space
        self.dim = dim
        self.max_elements = max_elements
        self.ef = ef
        self.M = M
        self.path = path
        self.open()

    def save(self):
        print("Saving Index")
        index_path = os.path.splitext(self.path)[0] + ".index"
        print("index_path", index_path)
        self.index.save_index(index_path)
        print("Index Saved")

    def close(self):
        self.save()
        self.db.close()
        self.graph = None
        self.index = None
        self.db = None

    def open(self):
        self.db = plyvel.DB(self.path, create_if_missing=True)
        
        index_path = os.path.splitext(self.path)[0] + ".index"
        print("index_path", index_path)
        self.index = hnswlib.Index(space=self.space, dim=self.dim) 
        if os.path.isfile(index_path):
            print("MemoryGraph: loading index")
            self.index.load_index(index_path)
        else:
            print("MemoryGraph: creating index")
            self.index.init_index(max_elements=self.max_elements, ef_construction=self.ef, M=self.M)
        self.index.set_ef(self.ef)

        print("MemoryGraph: loading graph")
        
        self.graph = nx.Graph()
        edges = self.load_all_edges()
        for from_node_id, to_node_id in edges:
            self.graph.add_edge(from_node_id, to_node_id)

        print("MemoryGraph: loaded", self.index.get_current_count(), "nodes", self.graph.number_of_edges(), "edges")


    def increment_node_id(self, count):
        return self.increment_id(count, b'd:n')

    def increment_observation_id(self, count):
        return self.increment_id(count, b'd:o')

    def increment_id(self, count, key):
        b = self.db.get(key)
        if b is None:
            node_id = count
        else:
            node_id = struct.unpack('>I', b)[0]
            node_id += count
        b = struct.pack('>I', node_id)
        self.db.put(key, b)
        return list(range(node_id-count+1, node_id+1))


    ######################
    # NODES
    ######################

    @staticmethod
    def numpy_to_bytes(a):
        return a.tobytes()

    @staticmethod
    def numpy_from_bytes(b):
        return np.frombuffer(b, dtype=np.float32)

    # node:[node_id] -> [node_data]
    @staticmethod
    def encode_node(node):
        return MemoryGraph.numpy_to_bytes(node["f"])

    @staticmethod
    def decode_node(k, v):
        node = dict()
        node["f"] = MemoryGraph.numpy_from_bytes(v)
        node["id"] = struct.unpack_from('>I', k, offset=1)[0]
        return node

    @staticmethod
    def node_key(node_id):
        return b'n' + struct.pack('>I', node_id)

    def get_node(self, node_id):
        return self.get_nodes([node_id])[0]

    def insert_node(self, node):
        return self.insert_nodes([node])[0]

    # def load_all_nodes(self):
    #     start = MemoryGraph.node_key(0)
    #     stop = MemoryGraph.node_key(4294967295)
    #     return [MemoryGraph.decode_node(key, value) for key, value in self.db.iterator(start=start, stop=stop)]

    def get_nodes(self, node_ids):
        return [{"f":f} for f in self.index.get_items(node_ids)]

    def insert_nodes(self, nodes):
        node_ids = self.increment_node_id(len(nodes))
        self.index.add_items([n["f"] for n in nodes], node_ids)
        for node_id in node_ids:
            self.graph.add_node(node_id)
        return node_ids


    ######################
    # EDGES
    ######################

    @staticmethod
    def edge_key(edge):
        return b'e' + struct.pack('>I', edge[0]) + struct.pack('>I', edge[1])

    def save_edges(self, edges):
        wb = self.db.write_batch()
        for from_node_id, to_node_id in edges:
            wb.put(MemoryGraph.edge_key((from_node_id, to_node_id)), b'')
        wb.write()

    def load_all_edges(self):
        start = MemoryGraph.edge_key((0,0))
        stop = MemoryGraph.edge_key((4294967295, 4294967295))
        return [(struct.unpack_from('>I', b, offset=1)[0], struct.unpack_from('>I', b, offset=5)[0]) for b in self.db.iterator(start=start, stop=stop, include_value=False)]


    ######################
    # OBSERVATIONS
    ######################

    # obs:[observation_id] -> [observation_data]
    @staticmethod
    def observation_key(observation_id):
        return b'o' + struct.pack('>I', observation_id)

    # get observation - observation is a dictionary
    def get_observation(self, observation_id):
        b = self.db.get(MemoryGraph.observation_key(observation_id))
        return json.loads(b.decode("utf-8") )

    def insert_observation(self, observation):
        return self.insert_observations([observation])[0]

    def get_observations(self, observation_ids):
        return [self.get_observation(observation_id) for observation_id in observation_ids]

    def insert_observations(self, observations):
        observation_ids = self.increment_observation_id(len(observations))
        wb = self.db.write_batch()
        for observation_id, observation in zip(observation_ids, observations):
            j = json.dumps(observation)
            b = j.encode("utf-8")
            wb.put(MemoryGraph.observation_key(observation_id), b)
        wb.write()
        return observation_ids

    # integrated_observation:[node_id]:[observation_id]
    @staticmethod
    def integrated_observations_key(node_id, observation_id):
        return b'i' + struct.pack('>I', node_id) + struct.pack('>I', observation_id)

    # observations that are integrated into node's features
    def get_integrated_observations(self, node_id):
        start = MemoryGraph.integrated_observations_key(node_id, 0)
        stop = MemoryGraph.integrated_observations_key(node_id, 4294967295)
        return [struct.unpack_from('>I', b, offset=5)[0] for b in self.db.iterator(start=start, stop=stop, include_value=False)]

    def add_integrated_observations(self, node_ids, observation_ids):
        wb = self.db.write_batch()
        for node_id, observation_id in zip(node_ids, observation_ids):
            wb.put(MemoryGraph.integrated_observations_key(node_id, observation_id), b'')
            wb.put(MemoryGraph.integrated_nodes_key(observation_id, node_id), b'')
        wb.write()


    # predicted_observation:[node_id]:[observation_id]
    @staticmethod
    def predicted_observations_key(node_id, observation_id):
        return b'p' + struct.pack('>I', node_id) + struct.pack('>I', observation_id)

    # observations that were predicted by node
    def get_predicted_observations(self, node_id):
        start = MemoryGraph.predicted_observations_key(node_id, 0)
        stop = MemoryGraph.predicted_observations_key(node_id, 4294967295)
        return [struct.unpack_from('>I', b, offset=5)[0] for b in self.db.iterator(start=start, stop=stop, include_value=False)]

    def add_predicted_observations(self, node_ids, observation_ids):
        wb = self.db.write_batch()
        for node_id, observation_id in zip(node_ids, observation_ids):
            wb.put(MemoryGraph.predicted_observations_key(node_id, observation_id), b'')
            wb.put(MemoryGraph.predicted_nodes_key(observation_id, node_id), b'')
        wb.write()


    # predicted_node:[observation_id]:[node_id]
    @staticmethod
    def predicted_nodes_key(observation_id, node_id):
        return b'q' + struct.pack('>I', observation_id) + struct.pack('>I', node_id)
    
    # nodes that predicted observation
    def get_predicted_nodes(self, observation_id):
        start = MemoryGraph.predicted_nodes_key(observation_id, 0)
        stop = MemoryGraph.predicted_nodes_key(observation_id, 4294967295)
        return [struct.unpack_from('>I', b, offset=5)[0] for b in self.db.iterator(start=start, stop=stop, include_value=False)]


    # integrated_node:[observation_id]:[node_id]
    @staticmethod
    def integrated_nodes_key(observation_id, node_id):
        return b'j' + struct.pack('>I', observation_id) + struct.pack('>I', node_id)

    # nodes that integrate observation
    def get_integrated_nodes(self, observation_id):
        start = MemoryGraph.integrated_nodes_key(observation_id, 0)
        stop = MemoryGraph.integrated_nodes_key(observation_id, 4294967295)
        return [struct.unpack_from('>I', b, offset=5)[0] for b in self.db.iterator(start=start, stop=stop, include_value=False)]


    def get_adjacencies(self, node_id, radius):
        return self._neighbors(node_id, radius)
        

    def _neighbors(self, v, radius, depth=0):
        result = set()
        for w in self.graph.neighbors(v):
            result.add(w)
            if depth + 1 < radius:
                result.update(self._neighbors(w, radius, depth+1))
        return result


    def insert_adjacency(self, from_id, to_id):
        self.save_edges([(from_id, to_id)])
        self.graph.add_edge(from_id, to_id)


    def knn_query(self, feats, k=1):
        return self.index.knn_query(feats, k)   


    def dnn_query(self, feature, distance):
        k = 100
        n = self.index.get_current_count()

        while True:
            if k > n:
                k = n

            _labels, _distances = self.index.knn_query([feature], k)  

            labels = _labels[0]
            distances = _distances[0]

            if distances[-1] > distance:
                idx = next(i for i,v in enumerate(distances) if v>distance)
                if idx == 0:
                    return [], []
                return labels[:(idx-1)], distances[:(idx-1)]

            if k == n:
                return labels, distances

            k = k * 2

            print("dnn expand", k)


    def index_count(self):
        return self.index.get_current_count()


    def random_walk(self, start, l, trials):
        visited = dict()

        for _ in range(trials):
            cur = start
            for _ in range(l):
                nei = list(self.graph.neighbors(cur))
                if len(nei) == 0:
                    break
                cur = random.choice(nei)
                if cur in visited:
                    visited[cur] += 1
                else:
                    visited[cur] = 1
    
        nodes = []
        count = []

        if not bool(visited):
            return [], []

        for key, value in visited.items():
            nodes.append(key)
            count.append(value)

        return zip(*sorted(zip(count, nodes), reverse=True))
    

    def get_community(self, node_id):
        counts, node_ids = self.random_walk(node_id, 10, 1000)

        n = 0
        for i in range(len(counts)):
            count = counts[i]
            if count < 200:
                break
            n += 1

        return node_ids[:n]


    # the goal here is to search through the set of all communities and find all the ones that have a 
    # max_pool distance within a range of the max_pool distance of the query community
    # candidate communities are ones that contain any member that is near any member of the quey community
    def search_group(self, features, feature_dis, community_dis, k=30):
        
        results = set()
        lab, dis = self.knn_query(features, k=k)
        features_max = np.max(features, axis=0)
        
        visited_nodes = set()

        for j in range(len(features)):
            labels = lab[j]
            distances = dis[j]

            for i in range(k):
                if distances[i] > feature_dis:
                    # break because distance are sorted and only increase from here
                    break
                label = labels[i]

                if label in visited_nodes:
                    print("label in visited_nodes")
                    continue
                visited_nodes.add(label)

                community = self.get_community(label)
                print("len(community)", len(community))
                if len(community) == 0:
                    continue
                community_features = np.array([self.get_node(c)["f"] for c in community])
                community_features_max = np.max(community_features, axis=0)
                d = spatial.distance.cosine(community_features_max, features_max)
                print("distance", d)
                if d <= community_dis:
                    results.add(frozenset(community))

        return results

    def distance(self, a, b):
        if self.space == 'cosine':
            n = preprocessing.normalize([a, b])
            return spatial.distance.cosine(n[0], n[1])
        else:
            return np.linalg.norm(a-b)


def resize_frame(image):
    return image


def get_rad_grid(g_pos, rad, shape, stride):

    top_left = (g_pos[0]-rad, g_pos[1]-rad)
    g_width = math.floor((shape[0] - 32)/stride)
    g_height = math.floor((shape[1] - 32)/stride)

    res = []

    for i in range(2*rad+1):
        p = (top_left[0]+i, top_left[1])
        if p[0] >= 0 and p[1] >= 0 and p[0] < g_width and p[1] < g_height:
            res.append(p)
 
    for i in range(2*rad+1):
        p = (top_left[0]+i, top_left[1]+(2*rad+1))
        if p[0] >= 0 and p[1] >= 0 and p[0] < g_width and p[1] < g_height:
            res.append(p)

    for i in range(2*rad-1):
        p = (top_left[0], top_left[1]+(i+1))
        if p[0] >= 0 and p[1] >= 0 and p[0] < g_width and p[1] < g_height:
            res.append(p)

    for i in range(2*rad-1):
        p = (top_left[0]+(2*rad), top_left[1]+(i+1))
        if p[0] >= 0 and p[1] >= 0 and p[0] < g_width and p[1] < g_height:
            res.append(p)

    #print(rad, g_pos, res)
    return res



def first_pos(kp_grid):
    ## TODO: if there are no key points in frame
    loc = random.choice(list(kp_grid.keys()))
    return loc, random.choice(kp_grid[loc])



def next_pos_play(kp_grid, shape, g_pos, stride):
    rad_grid = get_rad_grid(g_pos, 1, shape, stride)
    print("rad_grid", rad_grid)
    candidates = []

    for loc in rad_grid:

        if loc in kp_grid:
            candidates.append(loc)


    if len(candidates) == 0:
        return None, None

    loc = random.choice(candidates)

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



def extract_windows(frame, pos, window_size, walker_count):
    windows = np.empty((walker_count, window_size, window_size, 3))

    for i in range(walker_count):
        windows[i] = extract_window(frame, pos[i], window_size)

    return windows



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



def key_point_grid(orb, frame, stride):

    kp = orb.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)

    grid = dict()

    grid_offset_x = ((frame.shape[0] - 32) % stride)/2.0 + 16
    grid_offset_y = ((frame.shape[1] - 32) % stride)/2.0 + 16

    for k in kp:
        p = (k.pt[1],k.pt[0])
        g = (int(math.floor((p[0]-grid_offset_x)/stride)), int(math.floor((p[1]-grid_offset_y)/stride)))
        if g in grid:
            grid[g].append(p)
        else:
            grid[g] = [p]

    return grid



def paint_windows(positions, windows, frame, window_size, rect=-1):
    for i in range(len(positions)):
        pos = positions[i]
        x1 = int(round(pos[1] - window_size/2.0))
        x2 = x1 + window_size
        y1 = int(round(pos[0] - window_size/2.0))
        y2 = y1 + window_size
        
        window = windows[i]

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


def show_patches(path_windows, path_features, path_positions, frame_shape, memory_graph, window_size):
    print("show_patches")

    cv2.namedWindow("patches")

    frame = np.zeros((frame_shape[0], frame_shape[1], 3), np.uint8)

    paint_windows(path_positions, path_windows, frame, window_size, 0)

    # features, feature_dis, community_dis, k=30
    groups = list(memory_graph.search_group(path_features, .2, .2, 30))

    print("groups", groups)

    for i in range(len(groups)):
        group = list(groups[i])
        

        # node_ids = memory_graph.get_nodes(group)
        
        observation_ids = []
        for node_id in group:
            # print("node_id", node_id)
            integrated_observations = memory_graph.get_integrated_observations(node_id)
            observation_ids.extend(integrated_observations)
            predicted_observations = memory_graph.get_predicted_observations(node_id)
            observation_ids.extend(predicted_observations)

        windows = np.array([cv2.imread('./patches/patch'+str(observation_id)+'.png') for observation_id in observation_ids])

        observations = memory_graph.get_observations(observation_ids)

        positions = [(obs["y"], obs["x"]) for obs in observations]

        paint_windows(positions, windows, frame, window_size, i+1)

    cv2.imshow('patches', frame) 



def play_video(db_path, playback_random_walk_length = 10, window_size = 32, stride = 16, max_elements=10000000):

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONUP:
            return
        
        # kp = clostest_key_points(key_points, (x,y), 1)[0]

        res_frame = resize_frame(frame)
        kp_grid = key_point_grid(orb, res_frame, stride)
        print("len(kp_grid)", len(kp_grid))

        pos = (y, x)

        grid_offset_x = ((frame.shape[0] - 32) % stride)/2.0 + 16
        grid_offset_y = ((frame.shape[1] - 32) % stride)/2.0 + 16
        g_pos = (int(math.floor((pos[0]-grid_offset_x)/stride)), int(math.floor((pos[1]-grid_offset_y)/stride)))

        print("g_pos", g_pos)
        path = []

        for i in range(playback_random_walk_length):
            g_pos, pos = next_pos_play(kp_grid, res_frame.shape, g_pos, stride)
            print("g_pos, pos", g_pos, pos)
            if g_pos is None:
                break
            path.append(pos)

        path = list(set(path))

        windows = np.array([extract_window(res_frame, p, window_size) for p in path])

        preprocess_input(windows)
        features = model.predict(windows)
        features = features.reshape((windows.shape[0], 512))
        
        print("windows.shape, feats.shape", windows.shape, features.shape)
        show_patches(windows, features, path, frame.shape, memory_graph, window_size)

    orb = cv2.ORB_create(nfeatures=100000, fastThreshold=7)

    memory_graph = MemoryGraph(db_path, max_elements=max_elements, space='cosine', dim=512)

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
    

    cap.release() 
    cv2.destroyAllWindows() 



def build_graph(db_path, video_path, patch_dir, walk_length = 100, window_size = 32, stride = 16, runs = 1, max_files=None, max_frames=30*30, walker_count = 200, save_windows = True, max_elements=10000000):

    print("Starting...")

    if isdir(video_path):
        video_files = [f for f in listdir(video_path) if f.endswith('.mp4') and isfile(join(video_path, f))]
        random.shuffle(video_files)
        if max_files is not None:
            video_files = video_files[:max_files]
    else:
        sp = os.path.split(video_path)
        video_files = [sp[1]]
        video_path = sp[0]


    orb = cv2.ORB_create(nfeatures=100000, fastThreshold=7)

    # initialize VGG16
    model = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(32, 32, 3))

    # memory graph
    memory_graph = MemoryGraph(db_path, space='cosine', dim=512, max_elements=max_elements)
    memory_graph_walker = MemoryGraphWalker(memory_graph, distance_threshold = 0.10, identical_distance=0.01)
    
    # for each run though the video
    for r in range(runs):

        print("Run", r)

        video_file_count = 0

        for video_file in video_files:

            video_file_count += 1

            # open video file for a run though
            cap = cv2.VideoCapture(join(video_path, video_file))

            if save_windows:
                Path(join(patch_dir, video_file)).mkdir(parents=True, exist_ok=True)

            # walkers
            g_pos = [None for _ in range(walker_count)]
            pos = [None for _ in range(walker_count)]
            adj = [False for _ in range(walker_count)]

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

                kp_grid = key_point_grid(orb, frame, stride)

                for i in range(walker_count):
                    g_pos[i], pos[i], adj[i] = next_pos(kp_grid, frame.shape, g_pos[i], walk_length, stride)

                windows = extract_windows(frame, pos, window_size, walker_count)

                # extract cnn features from windows
                preprocess_input(windows)
                feats = model.predict(windows)
                feats = feats.reshape((windows.shape[0], 512))
        
                ids = memory_graph_walker.add_parrelell_observations(video_file, t, pos, adj, feats)

                restart_count = 0
                near_neighbor_count = 0
                is_identical_count = 0
                has_predictions_count = 0
                has_accurate_predictions_count = 0
                has_too_many_accurate_predictions_count = 0
                adjacencies_inserted = 0

                for i in range(walker_count):
                    if ids[i][0] is None:
                        # restart walk because we are in a very predictable spot
                        g_pos[i] = None
                        pos[i] = None
                        adj[i] = False  
                    if save_windows:
                        cv2.imwrite(join(patch_dir, video_file, 'patch_' + str(ids[i][1]) + '.png'), windows[i])


                    stats = ids[i][2]

                    if not stats["adj"]:
                        restart_count += 1
                    if "nearest_neighbor" in stats and stats["nearest_neighbor"] < .1:
                        near_neighbor_count += 1
                    if "predictions" in stats:
                        has_predictions_count += 1
                    if "accurate_predictions" in stats:
                        if stats["accurate_predictions"] > 0:
                            has_accurate_predictions_count += 1
                        if stats["accurate_predictions"] >= 10:
                            has_too_many_accurate_predictions_count += 1
                    if "identical" in stats and stats["identical"]:
                        is_identical_count += 1
                    if "adjacencies_inserted" in stats:
                        adjacencies_inserted += stats["adjacencies_inserted"]
                        
                print(
                    "vid", video_file_count, 
                    "frame", t+1,
                    "start", restart_count, 
                    "near", near_neighbor_count,
                    "iden", is_identical_count,
                    "pred", has_predictions_count,
                    "accu", has_accurate_predictions_count,
                    "many", has_too_many_accurate_predictions_count,
                    "adj", adjacencies_inserted,
                )
                
            cap.release()

    memory_graph.close()
    
    print("Done")

    
colors = [
    (1, 0, 103), (213, 255, 0), (255, 0, 86), (158, 0, 142), (14, 76, 161), (255, 229, 2), (0, 95, 57),(0, 255, 0), 
    (149, 0, 58), (255, 147, 126), (164, 36, 0), (0, 21, 68), (145, 208, 203), (98, 14, 0),(107, 104, 130), 
    (0, 0, 255), (0, 125, 181), (106, 130, 108), (0, 174, 126), (194, 140, 159), (190, 153, 112), (0, 143, 156), 
    (95, 173, 78), (255, 0, 0), (255, 0, 246), (255, 2, 157), (104, 61, 59), (255, 116, 163), (150, 138, 232), 
    (152, 255, 82), (167, 87, 64), (1, 255, 254), (255, 238, 232), (254, 137, 0), (189, 198, 255),(1, 208, 255), 
    (187, 136, 0), (117, 68, 177), (165, 255, 210), (255, 166, 254), (119, 77, 0), (122, 71, 130), (38, 52, 0), 
    (0, 71, 84), (67, 0, 44), (181, 0, 255), (255, 177, 103), (255, 219, 102), (144, 251, 146), (126, 45, 210), 
    (189, 211, 147), (229, 111, 254), (222, 255, 116), (0, 255, 120), (0, 155, 255), (0, 100, 1), (0, 118, 255), 
    (133, 169, 0), (0, 185, 23), (120, 130, 49), (0, 255, 198), (255, 110, 65), (232, 94, 190), (0, 0, 0)
]


# playback_random_walk_length = 10
# walk_length = 100
# window_size = 32
# stride = 16
# runs = 1
# max_frames=30*15
# walker_count = 200
# save_windows = True

# video_dir = './media/Tabletop Objects/videos/'
# db_path = "./data/tabletop_objects.db"




