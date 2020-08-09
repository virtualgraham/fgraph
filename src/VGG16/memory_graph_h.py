import hnswlib
import numpy as np
import networkx as nx
import random
from scipy import spatial
import time


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

    
    def add_parrelell_observations(self, t, pos, adj, feats):
        return [self.add_observation(t, pos[i], adj[i], feats[i], i) for i in range(len(feats))]


    def add_observation(self, t, pos, adj, feats, walker_id):

        print("\n-----------------------\n")
        print("Walker", walker_id, adj)
        
        l = d = None

        if self.memory_graph.index_count() >= self.knn:
            labels, distances = self.memory_graph.knn_query([feats], k = self.knn)
            l = labels[0]
            d = distances[0]

        if d is not None:
            print("Nearest Neighbor", d[0])

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

                f = pred['candidate_for_similar_to_curr']["f"]

                if self.memory_graph.space == 'cosine':
                    distance = spatial.distance.cosine(feats, f)
                else:
                    distance = np.linalg.norm(feats-f)

                if distance <= self.distance_threshold:
                    accurate_predictions.add(a)
                
                if len(accurate_predictions) >= self.accurate_prediction_limit:
                    print("Too many accurate_predictions")
                    break

                evaluated_ids.add(b)

            if len(predictions) > 0:
                print("Predictions", len(accurate_predictions), "of", len(predictions))
        
        print("frame", t, pos)

        if len(accurate_predictions) < self.accurate_prediction_limit:
            
            #if d is not None and sum([i < self.identical_distance for i in d]) > 3:
            if d is not None and (d[0] < self.identical_distance):
                oid = l[0]
                print("Using Identical")
            else:
                oid = self.memory_graph.insert_observation(t, pos[0], pos[1], feats)

            print("OID", oid)

            if adj:
                if walker_id in self.last_ids and self.last_ids[walker_id] is not None :
                    self.memory_graph.insert_adjacency(self.last_ids[walker_id], oid)

                for a in accurate_predictions:
                    self.memory_graph.insert_adjacency(a, oid)
        
        else:
            oid = None

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
                        props = self.memory_graph.get_observation(n)
                        self.predictions[walker_id].append(dict(id_similar_to_prev=label, id_similar_to_curr=n, candidate_for_similar_to_curr=props))

                    similar += 1

        self.last_ids[walker_id] = oid

        return oid
        


class MemoryGraph:

    def __init__(self, graph_path=None, index_path=None, space='cosine', dim=512, max_elements=1000000, ef=100, M=48):
        self.space = space
        self.dim = dim
        self.max_elements = max_elements
        self.ef = ef
        self.M = M

        #index   
        if index_path != None:
            self.load_index(index_path)
        else:
            self.init_index()

        #graph   
        if graph_path != None:
            self.load_graph(graph_path)
        else:
            self.init_graph()

        
    def save_index(self, index_path):
        self.index.save_index(index_path)


    def load_index(self, index_path):
        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.load_index(index_path)
        self.index.set_ef(self.ef)  


    def init_index(self):
        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.init_index(max_elements=self.max_elements, ef_construction=self.ef, M=self.M)
        self.index.set_ef(self.ef)  


    def save_graph(self, graph_path):
        nx.write_gpickle(self.graph, graph_path)


    def load_graph(self, graph_path):
        self.graph = nx.read_gpickle(graph_path)


    def init_graph(self):
        self.graph = nx.Graph(next_id=0)


    def get_observation(self, id):
        return self.graph.nodes[id]


    def get_observations(self, ids):
        return [self.get_observation(id) for id in ids]


    def insert_observation(self, t, y, x, f):
        id = self.graph.graph["next_id"]
        self.graph.graph["next_id"] = id + 1
        self.graph.add_node(id, t=t, y=y, x=x, f=f)
        self.index.add_items([f], [id])
        return id


    def get_adjacencies(self, id, radius):
        return self._neighbors(id, radius)
        

    def _neighbors(self, v, radius, depth=0):
        result = set()
        for w in self.graph.neighbors(v):
            result.add(w)
            if depth + 1 < radius:
                result.update(self._neighbors(w, radius, depth+1))
        return result


    def insert_adjacency(self, from_id, to_id):
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


    def random_walk(self, start, len, trials):
        visited = dict()

        for _ in range(trials):
            cur = start
            for _ in range(len):
                cur = random.choice(list(self.graph.neighbors(cur)))
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
    

    def search_group(self, features, cos_dis, depth, min_matches):
        results = set()
        for feature in features:
            labels, _ = self.dnn_query(feature, cos_dis)
            print("dnn count", len(labels))
            for label in labels:
                print("label", label)
                r = self.search_group_from_node(label, features, cos_dis, depth, min_matches)
                print("r", r)
                if r is not None:
                    results.add(frozenset(r))

        return results
    

    def search_group_from_node(self, node, features, cos_dis, depth, min_matches):
        edges = nx.bfs_edges(self.graph, node, depth_limit=depth)
        matches = set([node])
        feature_index_matched = set()
        
        # features contains the feature for the node so there will be a 
        for (_, v) in edges:
            f1 = self.get_observation(v)["f"]
            for i in range(len(features)):
                f2 = features[i]
                d = spatial.distance.cosine(f1, f2)
                if d <= cos_dis:
                   matches.add(v)
                   feature_index_matched.add(i)

        if len(feature_index_matched) >= min_matches:
            print("len(feature_index_matched)", len(feature_index_matched))
            return matches

        return None