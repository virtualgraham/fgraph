import hnswlib
import numpy as np
from graph_tool.all import Graph


knn = 10
distance_threshold = 1.50
identical_distance = 0.15 # probably the exact same frame and window 
adjacency_radius = 2


class MemoryGraphWalker:
    def __init__(self, memory_graph):
        self.memory_graph = memory_graph
        self.last_ids = dict()
        self.predictions = dict()

    
    def add_parrelell_observations(self, t, pos, feats):
        return [self.add_observation(t, pos[i], feats[i], i) for i in range(len(feats))]


    def add_observation(self, t, pos, feats, walker_id):

        print("Walker", walker_id)

        # if there is a feat that is almost identical use that id instead
        l = d = None

        if self.index.get_current_count() >= knn:
            labels, distances = self.index.knn_query([feats], k = knn)
            l = labels[0]
            d = distances[0]

        if d is not None:
            print(d[0])

        if d is not None and (d[0] < identical_distance):
            oid = l[0]
            print("Using Identical", oid, d[0])
        else:
            # Save Current Observation
            with self.driver.session() as session:
                oid = session.write_transaction(
                    insert_observation, 
                    t, 
                    pos[0], 
                    pos[1], 
                    feats_to_json(feats)
                )

        print("frame", t, oid, pos)

        # Save Previous -> Current Adjacency 
        if walker_id in self.last_ids:
            last_id = self.last_ids[walker_id]
            with self.driver.session() as session:
                session.write_transaction(
                    insert_adjacency, 
                    last_id,
                    oid,
                    0.0
                )

        # find correct predictions and reinforce with adjacency
        if walker_id in self.predictions:
            predictions = self.predictions[walker_id]

            accurate_predictions = 0

            for pred in predictions:
                f = np.array(json.loads(pred['candidate_for_similar_to_curr']["f"]))
                distance = np.linalg.norm(feats-f)
                
                if distance <= distance_threshold:
                    # add a link from prev to current
                    with self.driver.session() as session:
                        session.write_transaction(
                            insert_adjacency, 
                            pred["id_similar_to_prev"],
                            oid,
                            distance
                        )
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

            for n in range(knn):
                label = l[n]
                distance = d[n]

                if distance <= distance_threshold:
                    # Found a previous similar observation

                    # find other observations that have been seen near this one
                    with self.driver.session() as session:
                        next_adjacencies = session.write_transaction(
                            get_adjacencies, 
                            label,
                            adjacency_radius
                        )

                        for n in next_adjacencies:
                            self.predictions[walker_id].append(dict(id_similar_to_prev=label, candidate_for_similar_to_curr=n))

                    similar += 1



            if similar > 0:
                print("Similar ", similar)

        self.index.add_items([feats], [oid])
        self.last_ids[walker_id] = oid

        return oid
        


class MemoryGraph:
    # "my_graph.xml.gz"

    def __init__(self, index_path=None, graph_path=None):
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
        self.index = hnswlib.Index(space='l2', dim=256)
        self.index.load_index(index_path)
        self.index.set_ef(10)  


    def init_index(self):
        self.index = hnswlib.Index(space='l2', dim=256)
        self.index.init_index(max_elements=50000, ef_construction=100, M=16)
        self.index.set_ef(10)  


    def save_graph(self, graph_path):
        self.graph.save(graph_path)


    def load_graph(self, graph_path):
        self.graph = load_graph(graph_path)
        self.vertex_index = dict()
        for v in self.graph.vertices():
            self.vertex_index[self.graph.vp.id[v]] = v


    def init_graph(self):
        self.graph = Graph(directed=False)
        self.vertex_index = dict()
        self.graph.graph_properties["id"] = self.graph.new_graph_property("long")
        self.graph.graph_properties["id"] = 0
        self.graph.vertex_properties["id"] = self.graph.new_vertex_property("long")
        self.graph.vertex_properties["x"] = self.graph.new_vertex_property("double")
        self.graph.vertex_properties["y"] = self.graph.new_vertex_property("double")
        self.graph.vertex_properties["t"] = self.graph.new_vertex_property("long")
        self.graph.vertex_properties["f"] = self.graph.new_vertex_property("vector<double>")
        self.graph.edge_properties["d"] = self.graph.new_edge_property("double")


    def get_observation(self, id):
        v = self.vertex_index[id]
        return dict(id=id, x=self.graph.vp.x[v], y=self.graph.vp.y[v], t=self.graph.vp.t[v], f=self.graph.vp.f[v])


    def get_observations(self, ids):
        return [self.get_observation(id) for id in ids]


    def insert_observation(self, t, y, x, f):
        v = self.graph.add_vertex()

        id = self.graph.graph_properties["id"]
        self.graph.graph_properties["id"] = id + 1
        self.graph.vp.id[v] = id
        
        self.graph.vp.x[v] = x
        self.graph.vp.y[v] = y
        self.graph.vp.t[v] = t
        self.graph.vp.f[v] = f

        self.index.add_items([f], [id])
        return id


    def get_adjacencies(self, id, radius):
        v = self.vertex_index[id]
        return [self.vertex_index[n] for n in self._neighbors(v, radius)]
        

    def _neighbors(self, v, radius, depth=0):
        result = set()
        for w in v.out_neighbors():
            result.add(w)
            if depth + 1 < radius:
                result.update(self._neighbors(w, radius, depth+1))
        return result


    def insert_adjacency(self, from_id, to_id, distance):
        va = self.vertex_index[from_id]
        vb = self.vertex_index[to_id]
        e = self.graph.add_edge(va, vb)
        self.graph.ep.d[e] = distance


    def knn_query(self, feats, k=1):
        return self.index.knn_query(feats, k)   
   