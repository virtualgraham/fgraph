import math
import random
import time
import struct
import os.path
from os import listdir
from os.path import isfile, isdir, join, split, splitext
import json
from pathlib import Path
import threading
import re 
from itertools import chain

import numpy as np
import cv2
import hnswlib
# import networkx as nx
import plyvel
import community_walk_graph as cwg

from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.vgg16 import preprocess_input

# from collections import Counter

class MemoryGraphWalker:
    def __init__(self, memory_graph, knn = 30, accurate_prediction_limit = 10, distance_threshold = 0.1, adjacency_radius = 3, prediction_history_length=4, identical_distance=0.01):

        self.knn = knn
        self.accurate_prediction_limit = accurate_prediction_limit
        self.identical_distance = identical_distance
        self.distance_threshold = distance_threshold
        self.adjacency_radius = adjacency_radius
        self.prediction_history_length = prediction_history_length

        self.memory_graph = memory_graph
        self.last_ids = dict()
        self.predictions = dict()


    def add_parrelell_observations(self, file, t, pos, adj, feats, patches, objects, keep_times=False):
        if self.memory_graph.index_count() >= self.knn:
            labels, distances = self.memory_graph.knn_query(feats, k = self.knn)
        else:
            labels = [[] for i in range(len(feats))]
            distances = [[] for i in range(len(feats))]

        
        # print(labels)
        # get all the labels less than threshold distance together in one list
        labels_merged = list(chain.from_iterable(labels))
        distances_merged = list(chain.from_iterable(distances))

        neighbor_nodes_merged = list(set([l for l,d in zip(labels_merged, distances_merged) if d <= self.distance_threshold]))

        # TODO each community probably contains its own node, so probably it should be removed
        community_cache_list = self.memory_graph.get_communities(neighbor_nodes_merged, walk_trials=1000, member_portion=100)

        community_cache = dict([(neighbor_nodes_merged[i], community_cache_list[i]) for i in range(len(neighbor_nodes_merged))])

        return [self._add_observation(file, t, pos[i], adj[i], feats[i], patches[i], objects[i], labels[i], distances[i], i, community_cache, keep_times) for i in range(len(feats))]


    # def add_observation(self, file, t, pos, adj, feats, patch, obj, walker_id, keep_times=False):
    #     if self.memory_graph.index_count() >= self.knn:
    #         labels, distances = self.memory_graph.knn_query([feats], k = self.knn)

    #     return self._add_observation(file, t, pos, adj, feats, patch, obj, labels[0], distances[0], i, dict(), keep_times=keep_times)
        

    # TODO: should be parallelizable
    def _add_observation(self, file, t, pos, adj, feats, patch, obj, labels, distances, walker_id, community_cache, keep_times=False):
 
        stats = {"adj":adj}

        tm = TimeMarker(enabled=keep_times)

        observation = {"file":file, "t":t, "y":pos[0], "x":pos[1], "patch":patch}
        if obj is not None: observation["o"] = obj
        observation_id = self.memory_graph.insert_observation(observation)

        tm.mark(s="insert_observation")

        if len(distances) > 0 and distances[0] <= self.distance_threshold: 
            neighbor_nodes = set([l for l,d in zip(labels, distances) if d <= self.distance_threshold])
        else:
            neighbor_nodes = set()

        stats["near_neighbors_count"] = len(neighbor_nodes)
        # print("near_neighbors_count", len(neighbor_nodes))

        if len(distances) > 0:
            stats["nearest_neighbor"] = distances[0]

        tm.mark(l="find_correct_predictions", s="knn_query")

        accurate_predictions = set()
        evaluated_ids = set()

        # find correct predictions and reinforce with adjacency

        if adj and walker_id in self.predictions and len(neighbor_nodes) > 0:

            tm.mark(l="find_correct_predictions_inside")

            tm.mark(l="chain_predictions_into_set")
            ######### This is very expensive operation ##########
            predictions = set(chain.from_iterable(self.predictions[walker_id]))
            #########
            tm.mark(si="chain_predictions_into_set")

            # print(predictions)
            # if len(labels_set) == 0:
            #     stats["nearest_neighbor"]
            #     print("No neighbors in threshold distance")
            
            add_predicted_observations = set()
   
            ###################
            ###################
            tm.mark(l="build_accurate_predictions_set")

            for pred in predictions:
                if pred[1] in neighbor_nodes:
                    add_predicted_observations.add(pred[1]) # the node that is similar to the current observation
                    if len(accurate_predictions) < self.accurate_prediction_limit:
                        accurate_predictions.add(pred[0]) # the node that is similar to the previous observation
                    else:
                        break
            
            tm.mark(si="build_accurate_predictions_set")
            ###################
            ###################

            tm.mark(l="add_predicted_observations")

            if len(add_predicted_observations) > 0:
                self.memory_graph.add_predicted_observations(add_predicted_observations, [observation_id]*len(add_predicted_observations))

            tm.mark(si="add_predicted_observations")

            if len(predictions) > 0:
                stats["predictions"] = len(predictions)
                stats["accurate_predictions"] = len(accurate_predictions)
                # print("Predictions", len(accurate_predictions), "of", len(predictions))
        
            tm.mark(si="find_correct_predictions_inside")

        # print("frame", t, pos)
        tm.mark(si="find_correct_predictions")

        if len(accurate_predictions) < self.accurate_prediction_limit:
            
            #if d is not None and sum([i < self.identical_distance for i in d]) > 3:
            if len(distances) > 0 and (distances[0] < self.identical_distance):
                node_id = labels[0]
                stats["identical"] = True
                #print("Using Identical")
            else:
                node_id = self.memory_graph.insert_node({"f":feats})
            
            #print("NodeID", node_id)

            self.memory_graph.add_integrated_observations([node_id], [observation_id])
    
            stats["adjacencies_inserted"] = 0

            if adj:
                insert_adjacencies = []

                if walker_id in self.last_ids and self.last_ids[walker_id] is not None :
                    stats["adjacencies_inserted"] += 1
                    insert_adjacencies.append((self.last_ids[walker_id], node_id))
                    #self.memory_graph.insert_adjacency(self.last_ids[walker_id], node_id)

                for a in accurate_predictions:
                    stats["adjacencies_inserted"] += 1
                    insert_adjacencies.append((a, node_id))
                    #self.memory_graph.insert_adjacency(a, node_id)

                self.memory_graph.insert_adjacencies(insert_adjacencies)
        else:
            node_id = None

        tm.mark(s="insert_node_and_adjacencies")

        # make predictions
        if walker_id not in self.predictions:
            self.predictions[walker_id] = []

        h = self.predictions[walker_id]
        h.append(set())
        if len(h) > self.prediction_history_length:
            h.pop()

        # if self.memory_graph.index_count() >= self.knn len(neighbor_nodes) > 0:

        # neighbor_nodes_list = list()
        # multi_next_adjacencies = list()
        # neighbor_nodes_list_not_cached = list()

        # for label in neighbor_nodes:
        #     if label in community_cache:
        #         # print("cached")
        #         neighbor_nodes_list.append(label)
        #         multi_next_adjacencies.append(community_cache[label])
        #     else:
        #         # print("not cached")
        #         neighbor_nodes_list_not_cached.append(label)
        
        # communties_not_cached = self.memory_graph.get_communities(neighbor_nodes_list_not_cached, walk_trials=200, member_portion=20)

        # for i in range(len(neighbor_nodes_list_not_cached)):
        #     community_cache[neighbor_nodes_list_not_cached[i]] = communties_not_cached[i]

        # neighbor_nodes_list.extend(neighbor_nodes_list_not_cached)
        # multi_next_adjacencies.extend(communties_not_cached)
        
        for label in neighbor_nodes: #knn#
            next_adjacencies = community_cache[label]
            for n in next_adjacencies:
                # if n == label: continue
                self.predictions[walker_id][-1].add((label, n))

        tm.mark(s="make_predictions")

        if keep_times:
            stats["time"] = tm.saved

        self.last_ids[walker_id] = node_id

        return node_id, observation_id, stats
        
        
MAX_KEY_VALUE = 18446744073709551615

class MemoryGraph:
    #def __init__(self, path, space='cosine', dim=512, max_elements=1000000, ef=100, M=48, rebuild_index=False):
    def __init__(self, path, space='cosine', dim=512, max_elements=1000000, ef=300, M=64, rebuild_index=False):
        self.space = space
        self.dim = dim
        self.max_elements = max_elements
        self.ef = ef
        self.M = M
        self.path = path
        self.open(rebuild_index)

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

    def open(self, rebuild_index):
        self.db = plyvel.DB(self.path, create_if_missing=True)

        self.graph = cwg.new_graph()

        index_path = os.path.splitext(self.path)[0] + ".index"
        print("index_path", index_path)
        self.index = hnswlib.Index(space=self.space, dim=self.dim) 

        if os.path.isfile(index_path) and not rebuild_index:
            print("MemoryGraph: loading index")
            self.index.load_index(index_path)
            self.index.set_ef(self.ef)
            self.load_all_node_ids()
        else:
            print("MemoryGraph: building index")
            self.index.init_index(max_elements=self.max_elements, ef_construction=self.ef * 2, M=self.M)
            self.index.set_ef(self.ef)
            self.load_all_nodes()
            if cwg.len(self.graph) > 0:
                self.save()
        
        self.load_all_edges()

        print("MemoryGraph:", self.index.get_current_count(), "nodes")


    def load_all_node_ids(self):
        start = MemoryGraph.node_key(0)
        stop = MemoryGraph.node_key(MAX_KEY_VALUE)
        for key in self.db.iterator(start=start, stop=stop, include_value=False):
            cwg.add_node(self.graph, MemoryGraph.decode_node_key(key))


    def load_all_nodes(self):
        start = MemoryGraph.node_key(0)
        stop = MemoryGraph.node_key(MAX_KEY_VALUE)

        n = 0
        feats = []
        ids = []

        for key, value in self.db.iterator(start=start, stop=stop):
            node = MemoryGraph.decode_node(value) 
            node["id"] = MemoryGraph.decode_node_key(key)

            cwg.add_node(self.graph, node["id"])
            
            feats.append(node["f"])
            ids.append(node["id"])

            n += 1
            if n % 1000 == 0:
                print(n)
                self.index.add_items(feats, ids)
                feats = []
                ids = []

        if len(feats) > 0:
            self.index.add_items(feats, ids)


    def load_all_edges(self):
        print("MemoryGraph: loading graph")
        start = MemoryGraph.edge_key((0,0))
        stop = MemoryGraph.edge_key((MAX_KEY_VALUE, MAX_KEY_VALUE))

        for b in self.db.iterator(start=start, stop=stop, include_value=False):
            from_node_id = struct.unpack_from('>Q', b, offset=1)[0]
            to_node_id = struct.unpack_from('>Q', b, offset=9)[0]
            cwg.add_edge(self.graph, from_node_id, to_node_id)


    #######################################################

    def generate_node_ids(self, count):
        return [self.generate_id(MemoryGraph.node_key) for _ in range(count)]

    def generate_observation_ids(self, count):
        return [self.generate_id(MemoryGraph.observation_key) for _ in range(count)]

    def generate_id(self, key_fn):
        while True:
            id = random.getrandbits(64)
            b = self.db.get(key_fn(id))
            if b is None:
                return id

    ######################################################


    ######################
    # NODES
    ######################

    @staticmethod
    def numpy_to_bytes(a):
        return a.tobytes()

    @staticmethod
    def numpy_from_bytes(b):
        return np.frombuffer(b, dtype=np.float32)


    @staticmethod
    def community_key(node_id, walk_length, walk_trials, member_portion):
        return b'y' + struct.pack('>Q', node_id) + b':' + struct.pack('>I', walk_length) + b':' + struct.pack('>I', walk_trials) + b':' + struct.pack('>I', member_portion)

    @staticmethod
    def encode_community(community):
        b = bytearray()
        for c in community:
            b.extend(struct.pack('>Q', c))
        return bytes(b)

    @staticmethod
    def decode_community(b):
        return [struct.unpack_from('>Q', b, offset=i*8)[0] for i in range(int(len(b)/8))]

    # node:[node_id] -> [node_data]
    @staticmethod
    def encode_node(node):
        b = MemoryGraph.numpy_to_bytes(node["f"])
        # if "c" in node:
        #    b +=  MemoryGraph.encode_community(node["c"])
        return b

    @staticmethod
    def decode_node_key(k):
        return struct.unpack_from('>Q', k)[0]

    @staticmethod
    def decode_node(v):
        node = dict()
        node["f"] = MemoryGraph.numpy_from_bytes(v[0:(4*512)])
        # if len(v) > 4*512:
        #     node["c"] =  MemoryGraph.decode_community(v[4*512:])
        return node

    @staticmethod
    def decode_node_key(k):
        return struct.unpack_from('>Q', k, offset=1)[0]

    @staticmethod
    def node_key(node_id):
        return b'n' + struct.pack('>Q', node_id)

    def get_node(self, node_id):
        return self.get_nodes([node_id])[0]

    def insert_node(self, node):
        return self.insert_nodes([node])[0]

    def get_nodes(self, node_ids):
        return [{"f":f} for f in self.index.get_items(node_ids)]

    # TODO: should be parallelizable safe (plyvel, hnswlib, networkx)
    def insert_nodes(self, nodes):
        node_ids = self.generate_node_ids(len(nodes))

        wb = self.db.write_batch()
        for node_id, node in zip(node_ids, nodes):
            cwg.add_node(self.graph, node_id)
            wb.put(MemoryGraph.node_key(node_id), MemoryGraph.encode_node(node))
        wb.write()

        self.index.add_items([n["f"] for n in nodes], node_ids)

        return node_ids

    def read_node(self, node_id):
        b = self.db.get(MemoryGraph.node_key(node_id))
        if b is None:
            return None
        else:
            node = MemoryGraph.decode_node(b)
            node["id"] = node_id
            return node

    def write_node(self, node):
        self.db.put(MemoryGraph.node_key(node["id"]), MemoryGraph.encode_node(node))

    def read_community(self, node_id, walk_length, walk_trials, member_portion):
        b = self.db.get(MemoryGraph.community_key(node_id, walk_length, walk_trials, member_portion))
        if b is None:
            return None
        return MemoryGraph.decode_community(b)

    def write_community(self, node_id, walk_length, walk_trials, member_portion, community):
        self.db.put(MemoryGraph.community_key(node_id, walk_length, walk_trials, member_portion), MemoryGraph.encode_community(community))

    ######################
    # EDGES
    ######################

    @staticmethod
    def edge_key(edge):
        return b'e' + struct.pack('>Q', edge[0]) + struct.pack('>Q', edge[1])

    # TODO: should be parallelizable safe (plyvel)
    def save_edges(self, edges):
        wb = self.db.write_batch()
        for edge in edges:
            wb.put(MemoryGraph.edge_key(edge), b'')
        wb.write()


    ######################
    # Counts
    ######################

    @staticmethod
    def pixel_object_count_key(obj):
        return b'c:p:' + obj.encode()

    @staticmethod
    def pixel_count_key():
        return b'c:p'

    @staticmethod
    def frame_object_count_key(obj):
        return b'c:f:' + obj.encode()

    @staticmethod
    def frame_count_key():
        return b'c:f'

    @staticmethod
    def observation_object_count_key(obj):
        return b'c:o:' + obj.encode()

    @staticmethod
    def observation_count_key():
        return b'c:o'

    @staticmethod
    def video_object_count_key(obj):
        return b'c:v:' + obj.encode()

    @staticmethod
    def video_count_key():
        return b'c:v'

    def increment_count_wb(self, wb, key, amount):
        c = self.get_count(key)
        wb.put(key, struct.pack('>Q', c + amount))

    def get_count(self, key):
        c = self.db.get(key)
        if c is None:
            return 0
        else:
            return struct.unpack_from('>Q', c)[0]

    def get_counts(self):
        observation_count = self.get_count(MemoryGraph.observation_count_key())
        observation_objects = dict()
        for k,v in self.db.iterator(start=b'c:o:', stop=b'c:o:~'):
            observation_objects[k.decode().split(':')[2]] = struct.unpack_from('>Q', v)[0]

        frame_count = self.get_count(MemoryGraph.frame_count_key())
        frame_objects = dict()
        for k,v in self.db.iterator(start=b'c:f:', stop=b'c:f:~'):
            frame_objects[k.decode().split(':')[2]] = struct.unpack_from('>Q', v)[0]

        video_count = self.get_count(MemoryGraph.video_count_key())
        video_objects = dict()
        for k,v in self.db.iterator(start=b'c:v:', stop=b'c:v:~'):
            video_objects[k.decode().split(':')[2]] = struct.unpack_from('>Q', v)[0]

        pixel_count = self.get_count(MemoryGraph.pixel_count_key())
        pixel_objects = dict()
        for k,v in self.db.iterator(start=b'c:p:', stop=b'c:p:~'):
            pixel_objects[k.decode().split(':')[2]] = struct.unpack_from('>Q', v)[0]

        return {
            "observation_count": observation_count,
            "observation_objects": observation_objects,
            "frame_count": frame_count,
            "frame_objects": frame_objects,
            "video_count": video_count,
            "video_objects": video_objects,
            "pixel_count": pixel_count,
            "pixel_objects": pixel_objects,
        }


    # objects: a set of object names
    def increment_video_counts(self, objects):
        wb = self.db.write_batch()
        for obj in objects:
            self.increment_count_wb(wb, MemoryGraph.video_object_count_key(obj), 1)
        self.increment_count_wb(wb, MemoryGraph.video_count_key(), 1)
        wb.write()

    # object_pixels: a dict of object names -> pixels in object
    def increment_frame_counts(self, pixels, object_pixels):
        wb = self.db.write_batch()
        for obj, pix in object_pixels.items():
            self.increment_count_wb(wb, MemoryGraph.frame_object_count_key(obj), 1)
            self.increment_count_wb(wb, MemoryGraph.pixel_object_count_key(obj), pix)
        self.increment_count_wb(wb, MemoryGraph.frame_count_key(), 1)
        self.increment_count_wb(wb, MemoryGraph.pixel_count_key(), pixels)
        wb.write()


    ######################
    # OBSERVATIONS
    ######################
    # {"file":file, "t":t, "y":y, "x":x, "patch":patch}

    @staticmethod
    def encode_observation(observation):
        bt = struct.pack('>I', observation["t"]) # 4 bytes
        by = struct.pack('>d', observation["y"]) # 8 bytes
        bx = struct.pack('>d', observation["x"]) # 8 bytes
        #bpatch = observation["patch"].tobytes() # 3072 bytes

        if "o" in observation and observation["o"] is not None:
            bo = observation["o"].encode()
        else:
            bo = b''

        bolen = struct.pack('>H', len(bo)) # 2 bytes
        
        bfile = observation["file"].encode()
        bfilelen = struct.pack('>H', len(bfile)) # 2 bytes
        
        return bt + by + bx + bolen + bo + bfilelen + bfile


    @staticmethod
    def decode_observation(b):
        observation = dict()
        observation["t"] = struct.unpack_from('>I', b, offset=0)[0]
        observation["y"] = struct.unpack_from('>d', b, offset=4)[0]
        observation["x"] = struct.unpack_from('>d', b, offset=12)[0]
        # observation["patch"] = np.frombuffer(b[20:3092], dtype=np.uint8).reshape(32, 32, 3)

        offset = 20
        olen = struct.unpack_from('>H', b, offset=offset)[0]
        offset += 2
        if olen > 0:
            observation["o"] = b[offset:offset+olen].decode()
        offset += olen
        filelen = struct.unpack_from('>H', b, offset=offset)[0]
        offset += 2
        observation["file"] = b[offset:offset+filelen].decode()

        return observation


    # obs:[observation_id] -> [observation_data]
    @staticmethod
    def observation_key(observation_id):
        return b'o' + struct.pack('>Q', observation_id)

    # get observation - observation is a dictionary
    def get_observation(self, observation_id):
        b = self.db.get(MemoryGraph.observation_key(observation_id))
        observation = MemoryGraph.decode_observation(b)
        observation["id"] = observation_id
        return observation

    def insert_observation(self, observation):
        return self.insert_observations([observation])[0]

    def get_observations(self, observation_ids):
        return [self.get_observation(observation_id) for observation_id in observation_ids]

    # TODO: each observation should have a list of the objects that where in the frame
    
    # TODO: should be parallelizable safe (plyvel)
    def insert_observations(self, observations):
        observation_ids = self.generate_observation_ids(len(observations))
        wb = self.db.write_batch()

        for observation_id, observation in zip(observation_ids, observations):
            b = MemoryGraph.encode_observation(observation)
            wb.put(MemoryGraph.observation_key(observation_id), b)
            if "o" in observation and observation["o"] is not None:
                self.increment_count_wb(wb, MemoryGraph.observation_object_count_key(observation["o"]), 1)

        self.increment_count_wb(wb, MemoryGraph.observation_count_key(), len(observations))
        
        wb.write()
        return observation_ids

    # integrated_observation:[node_id]:[observation_id]
    @staticmethod
    def integrated_observations_key(node_id, observation_id):
        return b'i' + struct.pack('>Q', node_id) + struct.pack('>Q', observation_id)

    # observations that are integrated into node's features
    def get_integrated_observations(self, node_id):
        start = MemoryGraph.integrated_observations_key(node_id, 0)
        stop = MemoryGraph.integrated_observations_key(node_id, MAX_KEY_VALUE)
        return [struct.unpack_from('>Q', b, offset=9)[0] for b in self.db.iterator(start=start, stop=stop, include_value=False)]

    # TODO: should be parallelizable safe (plyvel)
    def add_integrated_observations(self, node_ids, observation_ids):
        wb = self.db.write_batch()
        for node_id, observation_id in zip(node_ids, observation_ids):
            wb.put(MemoryGraph.integrated_observations_key(node_id, observation_id), b'')
            wb.put(MemoryGraph.integrated_nodes_key(observation_id, node_id), b'')
        wb.write()


    # predicted_observation:[node_id]:[observation_id]
    @staticmethod
    def predicted_observations_key(node_id, observation_id):
        return b'p' + struct.pack('>Q', node_id) + struct.pack('>Q', observation_id)

    # observations that were predicted by node
    def get_predicted_observations(self, node_id):
        start = MemoryGraph.predicted_observations_key(node_id, 0)
        stop = MemoryGraph.predicted_observations_key(node_id, MAX_KEY_VALUE)
        return [struct.unpack_from('>Q', b, offset=9)[0] for b in self.db.iterator(start=start, stop=stop, include_value=False)]

    # TODO: should be parallelizable safe (plyvel)
    def add_predicted_observations(self, node_ids, observation_ids):
        wb = self.db.write_batch()
        for node_id, observation_id in zip(node_ids, observation_ids):
            wb.put(MemoryGraph.predicted_observations_key(node_id, observation_id), b'')
            wb.put(MemoryGraph.predicted_nodes_key(observation_id, node_id), b'')
        wb.write()


    # predicted_node:[observation_id]:[node_id]
    @staticmethod
    def predicted_nodes_key(observation_id, node_id):
        return b'q' + struct.pack('>Q', observation_id) + struct.pack('>Q', node_id)
    
    # nodes that predicted observation
    def get_predicted_nodes(self, observation_id):
        start = MemoryGraph.predicted_nodes_key(observation_id, 0)
        stop = MemoryGraph.predicted_nodes_key(observation_id, MAX_KEY_VALUE)
        return [struct.unpack_from('>Q', b, offset=9)[0] for b in self.db.iterator(start=start, stop=stop, include_value=False)]


    # integrated_node:[observation_id]:[node_id]
    @staticmethod
    def integrated_nodes_key(observation_id, node_id):
        return b'j' + struct.pack('>Q', observation_id) + struct.pack('>Q', node_id)

    # nodes that integrate observation
    def get_integrated_nodes(self, observation_id):
        start = MemoryGraph.integrated_nodes_key(observation_id, 0)
        stop = MemoryGraph.integrated_nodes_key(observation_id, MAX_KEY_VALUE)
        return [struct.unpack_from('>Q', b, offset=9)[0] for b in self.db.iterator(start=start, stop=stop, include_value=False)]


    def get_adjacencies(self, node_id, radius):
        cwg.neighbors(self.graph, node_id, radius)
   
   
    #     return self._neighbors(node_id, radius, set())
        

    # def _neighbors(self, v, radius, path):
    #     result = set()
    #     for w in cwg.neighbors(self.graph, v, 1):
    #         if w in path:
    #             continue
    #         result.add(w)
    #         if len(path) + 1 < radius:
    #             result.update(self._neighbors(w, radius, path.union({w})))
    #     return result

    # TODO: should be parallelizable safe (networkx)
    def insert_adjacency(self, from_id, to_id):
        self.save_edges([(from_id, to_id)])
        cwg.add_edge(self.graph, from_id, to_id)


    def insert_adjacencies(self, edges):
        self.save_edges(edges)
        for e in edges:
            cwg.add_edge(self.graph, e[0], e[1])

    # TODO: should be parallelizable safe (hnswlib)
    def knn_query(self, feats, k=1):
        if len(feats) == 0:
            return ([],[])
        return self.index.knn_query(feats, k)   

    # TODO: should be parallelizable safe (hnswlib)
    def index_count(self):
        return cwg.len(self.graph)
        # return self.index.get_current_count()


    # def random_walk(self, start, l, trials):
    #     visited = dict()

    #     for _ in range(trials):
    #         cur = start
    #         for _ in range(l):
    #             nei = list(cwg.neighbors(self.graph, cur, 1))
    #             if len(nei) == 0:
    #                 break
    #             cur = random.choice(nei)
    #             if cur in visited:
    #                 visited[cur] += 1
    #             else:
    #                 visited[cur] = 1
    
    #     nodes = []
    #     count = []

    #     if not bool(visited):
    #         return [], []

    #     for key, value in visited.items():
    #         nodes.append(key)
    #         count.append(value)

    #     return zip(*sorted(zip(count, nodes), reverse=True))
    

    def get_communities(self, node_ids, walk_length=10, walk_trials=1000, member_portion=200):
        return cwg.communities(self.graph, node_ids, walk_length, walk_trials, member_portion)


    def get_community(self, node_id, walk_length=10, walk_trials=1000, member_portion=200, save_to_db=True):
        
        if save_to_db:
            community = self.read_community(node_id, walk_length, walk_trials, member_portion)
            if community is not None:
                # print("read community")
                return set(community)

        # counts, node_ids = self.random_walk(node_id, walk_length, walk_trials)

        # n = 0
        # for i in range(len(counts)):
        #     count = counts[i]
        #     if count < member_portion:
        #         break
        #     n += 1

        # community = node_ids[:n]
        
        community = cwg.community(self.graph, node_id, walk_length, walk_trials, member_portion)

        if save_to_db:
            self.write_community(node_id, walk_length, walk_trials, member_portion, community)

        return set(community)



    def observations_for_nodes(self, node_ids):
        observation_ids = []
        for node_id in node_ids:
            integrated_observations = self.get_integrated_observations(node_id)
            observation_ids.extend(integrated_observations)
            predicted_observations = self.get_predicted_observations(node_id)
            observation_ids.extend(predicted_observations)
        return observation_ids


    from collections import Counter

    # the goal here is to search through the set of all communities and find all the ones that have a 
    # max_pool distance within a range of the max_pool distance of the query community
    # candidate communities are ones that contain any member that is near any member of the quey community
    def search_group(self, features, feature_dis=0.2, community_dis=0.2, k=30, walk_length=10, walk_trials=1000, member_portion=200):
        
        results = set()

        if len(features) == 0:
            return results

        lab, dis = self.knn_query(features, k=k)
        features_max = np.max(features, axis=0)
        
        visited_nodes = set()

        # degrees = []

        for j in range(len(features)):
            labels = lab[j]
            distances = dis[j]

            len_results = len(results)

            for i in range(k):
                if distances[i] > feature_dis:
                    # break because distance are sorted and only increase from here
                    break
                label = labels[i]
                
                if label in visited_nodes:
                    # print("label in visited_nodes")
                    continue
                visited_nodes.add(label)

                community = self.get_community(label, walk_length, walk_trials, member_portion)
                # print("len(community)", len(community))
                if len(community) == 0:
                    continue
                community_features = np.array([self.get_node(c)["f"] for c in community])
                community_features_max = np.max(community_features, axis=0)
                d = self.distance(community_features_max, features_max)
                # print("distance", d)
                if d <= community_dis:
                    results.add(frozenset(community))

            # print("found", len(results) - len_results, "communities")

        # print(Counter(degrees))

        return results


    def distance(self, a, b):
        if self.space == 'cosine':
            return np_cosine(a, b)
        else:
            return np.linalg.norm(a-b)


def np_cosine(x,y):
    return 1 - np.inner(x,y)/math.sqrt(np.dot(x,x)*np.dot(y,y))


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


def object_names_from_video_file(video_file):
    return re.findall('_([a-z]+)', video_file)


def pixel_counts(obj_frame, center_size):
    unique, counts = np.unique(obj_frame, return_counts=True)
    unique = [object_name_for_idx(o) for o in unique]

    min_pix = center_size * center_size * 0.9
    pixels = dict([(u, c) for u, c in zip(unique, counts) if u is not None and c > min_pix])
    
    return pixels


def extract_object(window, center_size):
    c = np.bincount(window.flatten())
    if np.max(c) >= center_size*center_size*.90:
        return object_name_for_idx(np.argmax(c))
    else:
        return None


def extract_objects(obj_frame, pos, center_size):
    windows = np.empty((len(pos), center_size, center_size), dtype=np.uint8)

    for i in range(len(pos)):
        windows[i] = extract_window(obj_frame, pos[i], center_size)

    return [extract_object(w, center_size) for w in windows]


def extract_windows(frame, pos, window_size):
    windows = np.empty((len(pos), window_size, window_size, 3), dtype=np.uint8)

    for i in range(len(pos)):
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

def extract_window_pixels(pos, frame_shape, window_size):
    half_w = window_size/2.0
    bottom_left = [int(round(pos[0]-half_w)), int(round(pos[1]-half_w))]
    top_right = [bottom_left[0]+window_size, bottom_left[1]+window_size]
   
    if bottom_left[0] < 0:
        top_right[0] -= bottom_left[0]
        bottom_left[0] = 0

    if bottom_left[1] < 0:
        top_right[1] -= bottom_left[1]
        bottom_left[1] = 0

    if top_right[0] >= frame_shape[0]:
        bottom_left[0] -= (top_right[0]-frame_shape[0]+1)
        top_right[0] = frame_shape[0]-1

    if top_right[1] >= frame_shape[1]:
        bottom_left[1] -= (top_right[1]-frame_shape[1]+1)
        top_right[1] = frame_shape[1]-1

    points = []
    for y in range(bottom_left[0], top_right[0]):
        for x in range(bottom_left[1], top_right[1]):
            points.append((y,x))
            
    return points

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

        observations = memory_graph.get_observations(observation_ids)

        windows = np.array([obs["patch"] for obs in observations])
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
        print("windows.shape, windows.dtype", windows.shape, windows.dtype)

        preprocess_input(windows)
        features = model.predict(windows)
        features = features.reshape((windows.shape[0], 512))
        
        print("windows.shape, windows.dtype", windows.shape, windows.dtype)
        print("feats.shape, feats.dtype", windows.shape, windows.dtype)

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



def build_graph(db_path, video_path, mask_path, video_files, walk_length = 100, window_size = 32, center_size = 16, stride = 16, runs = 1, max_frames=30*30, walker_count = 200, max_elements=10000000, keep_times=False):

    print("Starting...")

    t1 = TimeMarker(enabled=keep_times)

    random.shuffle(video_files)

    t1.mark(p="TIME video file paths")

    orb = cv2.ORB_create(nfeatures=100000, fastThreshold=7)

    t1.mark(p="TIME init orb")

    # initialize VGG16
    model = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(32, 32, 3))

    t1.mark(p="TIME init VGG16")

    # memory graph
    memory_graph = MemoryGraph(db_path, space='cosine', dim=512, max_elements=max_elements)
    memory_graph_walker = MemoryGraphWalker(memory_graph, distance_threshold = 0.10, identical_distance=0.01)
    
    t1.mark(p="TIME init Memory Graph")

    visited_videos = set()

    # for each run though the video
    for r in range(runs):

        print("Run", r)

        video_file_count = 0

        for video_file in video_files:
            
            print(video_file)
            
            visited_videos.add(video_file)

            t2 = TimeMarker(enabled=keep_times)

            video_file_count += 1

            # open video file for a run though
            mask = cv2.VideoCapture(join(mask_path,"mask_"+video_file))
            video = cv2.VideoCapture(join(video_path, video_file))

            # objects: a set of object names
            video_objects = object_names_from_video_file(video_file)
            print("video_objects", video_objects)
            memory_graph.increment_video_counts(video_objects)

            # walkers
            g_pos = [None for _ in range(walker_count)]
            pos = [None for _ in range(walker_count)]
            adj = [False for _ in range(walker_count)]

            done = False

            t2.mark(p="TIME open video")

            # for each frame
            for t in range(max_frames):
                if done:
                    break

                t3 = TimeMarker(enabled=keep_times)

                video_ret, video_frame = video.read()
                mask_ret, mask_frame = mask.read()
                
                t3.mark(p="TIME read video frame")

                if video_ret == False or mask_ret == False:
                    done = True
                    break

                t3.mark(p="TIME resize_frame")

                obj_frame = color_fun(mask_frame)

                kp_grid = key_point_grid(orb, video_frame, stride)

                t3.mark(p="TIME key_point_grid")

                for i in range(walker_count):
                    g_pos[i], pos[i], adj[i] = next_pos(kp_grid, video_frame.shape, g_pos[i], walk_length, stride)

                t3.mark(p="TIME walker_count x next_pos")

                patches = extract_windows(video_frame, pos, window_size)
                windows = patches.astype(np.float64)

                t3.mark(p="TIME extract_windows")

                # extract cnn features from windows
                preprocess_input(windows)
                feats = model.predict(windows)
                feats = feats.reshape((windows.shape[0], 512))
        
                t3.mark(p="TIME preprocess_input + model.predict")

                objects = extract_objects(obj_frame, pos, center_size)
                ids = memory_graph_walker.add_parrelell_observations(video_file, t, pos, adj, feats, patches, objects, keep_times)

                observations_with_objects = len([x for x in objects if x is not None])

                t3.mark(p="TIME add_parrelell_observations")

                restart_count = 0
                near_neighbor_count = 0
                is_identical_count = 0
                has_predictions_count = 0
                has_accurate_predictions_count = 0
                has_too_many_accurate_predictions_count = 0
                adjacencies_inserted = 0
                nn_gte_10 = 0
                nn_gte_20 = 0

                time_stats = dict()

                for i in range(walker_count):
                    if ids[i][0] is None:
                        # restart walk because we are in a very predictable spot
                        g_pos[i] = None
                        pos[i] = None
                        adj[i] = False  

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
                        if stats["accurate_predictions"] >= memory_graph_walker.accurate_prediction_limit:
                            has_too_many_accurate_predictions_count += 1
                    if "identical" in stats and stats["identical"]:
                        is_identical_count += 1
                    if "adjacencies_inserted" in stats:
                        adjacencies_inserted += stats["adjacencies_inserted"]
                    if "near_neighbors_count" in stats:
                        if stats["near_neighbors_count"] >= 10:
                            nn_gte_10 += 1
                        if stats["near_neighbors_count"] >= 20:
                            nn_gte_20 += 1
                    if keep_times:
                        for k, v in stats["time"].items():
                            if k not in time_stats:
                                time_stats[k] = v
                            else:
                                time_stats[k] += v

                t3.mark(p="TIME write patches + compute stats")

                object_pixels = pixel_counts(obj_frame, center_size)
                # print("object_pixels", object_pixels)
                memory_graph.increment_frame_counts(obj_frame.shape[0]*obj_frame.shape[1], object_pixels)

                if keep_times:
                    print(time_stats)

                print(
                    "vid", video_file_count, 
                    "frame", t+1,
                    "start", restart_count, 
                    "nn00", near_neighbor_count,
                    "nn10", nn_gte_10,
                    "nn20", nn_gte_20,
                    "iden", is_identical_count,
                    "pred", has_predictions_count,
                    "accu", has_accurate_predictions_count,
                    "many", has_too_many_accurate_predictions_count,
                    "obj", observations_with_objects,
                    "adj", adjacencies_inserted,
                )

                
                
            mask.release()
            video.release()



            # if r < (runs-1):
            #     memory_graph.save()

    counts = memory_graph.get_counts()
    print("counts", counts)

    memory_graph.close()
    
    print(visited_videos)

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


color_objects = [
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
]

def object_name_for_idx(idx):
    if idx > 0 and idx < len(color_objects):
        return color_objects[idx]
    return None

def color_fun(mask_frame):
    o = np.zeros((1280, 720), dtype=np.uint8)

    for i in range(1, len(color_vectors)):
        o[np.linalg.norm(color_full[i] - mask_frame, axis=2)<color_dist] = i
    
    return o


# utility for working with marks and intervals
# a mark is a nanosecond timestamp returned from time.time_ns()
# an interval is time elapsed between two marks
class TimeMarker:
    def __init__(self, enabled=True, l="start"):
        self.enabled = enabled
        if enabled:
            self.last = time.time_ns()
            self.mark_dict = {l: self.last}
            self.saved = {}

    # sets a new time mark and calculates the interval from the new time mark to a previous time mark
    # l: give this mark a label to be used later
    # i: calculate the interval since a labeled mark instead of using simply the last mark
    # s: save the interval in a dict with this name
    # a: add this interval to the value in saved dict 
    # p: print the interval with the given text
    # si: a shortcut to set s and i to the same value
    def mark(self, l=None, i=None, s=None, a=None, p=None, si=None):
        if not self.enabled:
            return 0

        if si is not None:
            s = si
            i = si

        t = time.time_ns()
        
        if l is not None:
            self.mark_dict[l] = t
        
        if i is not None:
            r = t - self.mark_dict[i]     
        else:
            r = t - self.last

        self.last = t

        if s is not None:
            self.saved[s] = r
        elif a is not None:
            if a not in self.saved:
                self.saved[a] = r
            else:
                self.saved[a] += r
        if p is not None:
            print(p, r)

        return r


    