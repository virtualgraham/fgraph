# Add Observation(s) for current time

# When an observation is added 
# an adjacency is created between previous observation and current
# if current prediction is similar to predictions, adjacencies are created betwen prediction predecessors and current

# when a observation is added all closely similar nodes and nodes within a radius are retrieved. These nodes are added to an in memory graph and marked with the current time.
# Nodes with time older than a given range are cleared from the in memory database

# Once an in memory graph has been updated, a community search is done

# In theory then we will have all the historical observations associated with recently observed objects in memory and partitioned by object

# similar observations to previous observation are retrieved. if any of their forward adjacencies are similar to current, adjacencies to those are added



import hnswlib
from neo4j import GraphDatabase
import json 
import numpy as np

knn = 10
distance_threshold = 1.50
identical_distance = 0.15 # probably the exact same frame and window 
adjacency_radius = 2

def feats_to_json(a):
    return json.dumps(a.tolist())


def json_to_feats(s):
    load = json.loads(s)
    return np.asarray(load)
    

def get_adjacencies(tx, id, radius):
    result = tx.run("MATCH (f:Observation)-[:ADJACENT*.." + str(radius) + "]-(t:Observation) WHERE ID(f) = $id RETURN collect(DISTINCT t)", id=int(id))
    r = result.single()
    return r["collect(DISTINCT t)"]


def get_observation_group(tx, id):
    result = tx.run("MATCH (o:Observation) WHERE ID(o) = $id RETURN o.group", id=int(id))
    r = result.single()
    return r["o.group"]


def insert_observation(tx, t, y, x, f):
    result = tx.run("Create(o:Observation {t:$t, y:$y, x:$x, f:$f}) RETURN ID(o)", t=int(t), y=float(y), x=float(x), f=f)
    r = result.single()
    return r["ID(o)"]


def insert_adjacency(tx, from_id, to_id, distance):
    tx.run( "MATCH (from:Observation) WHERE ID(from) = $from_id "
            "MATCH (to:Observation) WHERE ID(to) = $to_id "
            "MERGE (from)-[rel:ADJACENT {d: $distance}]->(to)", from_id=int(from_id), to_id=int(to_id), distance=float(distance))


def get_nodes(tx, ids):
    result = tx.run("MATCH (o:Observation) WHERE ID(o) IN $ids RETURN o", ids=ids)
    return [r["o"] for r in result]


def all_edges(tx):
    result = tx.run("Match (from:Observation)-[a:ADJACENT]->(to:Observation) Return a")
    return [r["a"] for r in result]


def db_all_edges(db="neo4j://localhost:7687", auth=("neo4j", "password")):
    driver = GraphDatabase.driver(db, auth=auth)

    with driver.session() as session:
        result = session.write_transaction(all_edges)

    driver.close()

    return result


def node_embeddings(db="neo4j://localhost:7687", auth=("neo4j", "password")):
    driver = GraphDatabase.driver(db, auth=auth)

    ids = []
    embeddings = []

    with driver.session() as session:
        result = session.run("CALL gds.alpha.node2vec.stream('graph')")

        for record in result:
            ids.append(record["nodeId"])
            embeddings.append(record["embedding"])

    driver.close()

    return embeddings, ids

def build_node_embeddings_index():
    embeddings, ids = node_embeddings()

    print("len(embeddings)", len(embeddings))

    index = hnswlib.Index(space='l2', dim=128)
    index.init_index(max_elements=len(embeddings), ef_construction=100, M=16)
    index.set_ef(10) 
    index.add_items(embeddings, ids)

    index.save_index("./data/node_embeddings_index.bin")


class MemoryGraph:
    def __init__(self, index_path=None, node_embeddings_path=None, db="neo4j://localhost:7687"):
        self.index_path = index_path
        self.node_embeddings_path = node_embeddings_path
        self.db = db
        self.last_ids = dict()
        self.predictions = dict()
        self.open()


    def open(self):
        #knn
        self.index = hnswlib.Index(space='l2', dim=256)
        
        if self.index_path != None:
            self.load_index()
        else:
            self.init_index()
        
        if self.node_embeddings_path != None:
            self.node_embeddings_index = hnswlib.Index(space='l2', dim=128)
            self.load_node_embeddings_index()

        self.index.set_ef(10)   

        #db
        self.driver = GraphDatabase.driver(self.db, auth=("neo4j", "password"))
        

    def save_index(self, index_path):
        self.index_path = index_path
        self.index.save_index(index_path)

    def load_index(self):
        self.index.load_index(self.index_path)

    def load_node_embeddings_index(self):
        self.node_embeddings_index.load_index(self.node_embeddings_path)

    def init_index(self):
        self.index.init_index(max_elements=50000, ef_construction=100, M=16)

    def close(self):
        self.driver.close()

    def knn_query(self, feats, k=1):
        return self.index.knn_query(feats, k)

    def knn_query_node_embeddings(self, node_id, k=1):
        feats = self.node_embeddings_index.get_items([node_id])
        return self.node_embeddings_index.knn_query(feats, k)

    def get_observation_group(self, label):
        with self.driver.session() as session:
            return session.read_transaction(get_observation_group, label)

    def get_nodes(self, ids):
        with self.driver.session() as session:
            return session.read_transaction(get_nodes, ids)

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