import hnswlib
import numpy as np
import networkx as nx
import random
from scipy import spatial


class MemoryGraphWalker:
    def __init__(self, memory_graph, knn = 10, distance_threshold = 1.0, identical_distance = 0.15, adjacency_radius = 2):

        self.knn = knn
        self.distance_threshold = distance_threshold
        self.identical_distance = identical_distance
        self.adjacency_radius = adjacency_radius

        self.memory_graph = memory_graph
        self.last_ids = dict()
        self.predictions = dict()

    
    def add_parrelell_observations(self, t, pos, feats):
        return [self.add_observation(t, pos[i], feats[i], i) for i in range(len(feats))]


    def add_observation(self, t, pos, feats, walker_id):

        print("Walker", walker_id)

        # if there is a feat that is almost identical use that id instead
        l = d = None

        if self.memory_graph.index_count() >= self.knn:
            labels, distances = self.memory_graph.knn_query([feats], k = self.knn)
            l = labels[0]
            d = distances[0]

        if d is not None:
            print("Nearest Neighbor", d[0])

        if d is not None and (d[0] < self.identical_distance):
            oid = l[0]
            print("Using Identical", oid, d[0])
        else:
            # Save Current Observation
            oid = self.memory_graph.insert_observation(t, pos[0], pos[1], feats)

        print("frame", t, oid, pos)

        # Save Previous -> Current Adjacency 
        if walker_id in self.last_ids:
            last_id = self.last_ids[walker_id]
            self.memory_graph.insert_adjacency(last_id, oid, 0.0)


        # find correct predictions and reinforce with adjacency
        if walker_id in self.predictions:
            predictions = self.predictions[walker_id]

            accurate_predictions = 0

            for pred in predictions:
                f = pred['candidate_for_similar_to_curr']["f"]

                if self.memory_graph.space == 'cosine':
                    distance = spatial.distance.cosine(feats, f)
                else:
                    distance = np.linalg.norm(feats-f)
                    
                #print("distance <= self.distance_threshold", distance,  self.distance_threshold)
                if distance <= self.distance_threshold:
                    # add a link from prev to current
                    self.memory_graph.insert_adjacency(pred["id_similar_to_prev"], oid, distance)
                    accurate_predictions += 1

                if accurate_predictions >= 10:
                    print("Too many accurate_predictions")
                    break

            if len(predictions) > 0:
                print("Predictions", accurate_predictions, "of", len(predictions))

        # make predictions
        self.predictions[walker_id] = []

        if l is not None and d is not None:

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
                        self.predictions[walker_id].append(dict(id_similar_to_prev=label, candidate_for_similar_to_curr=props))

                    similar += 1


            if similar > 0:
                print("Similar ", similar)

        self.last_ids[walker_id] = oid

        return oid
        


class MemoryGraph:

    def __init__(self, graph_path=None, index_path=None, space='l2', dim=256):
        self.space = space
        self.dim = dim
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
        self.index.set_ef(10)  


    def init_index(self):
        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.init_index(max_elements=50000, ef_construction=100, M=16)
        self.index.set_ef(10)  


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


    def insert_adjacency(self, from_id, to_id, distance):
        #print("insert_adjacency", from_id, to_id)
        self.graph.add_edge(from_id, to_id, d=distance)


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
                # print(d)
                if d <= cos_dis:
                   matches.add(v)
                   feature_index_matched.add(i)

        if len(feature_index_matched) >= min_matches:
            print("len(feature_index_matched)", len(feature_index_matched))
            return matches

        return None