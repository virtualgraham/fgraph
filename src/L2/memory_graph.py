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
distance_threshold = 1.5
identical_distance = 1.0

def feats_to_json(a):
    return json.dumps(a.tolist())


def json_to_feats(s):
    load = json.loads(s)
    return np.asarray(load)
    
def get_next_adjacencies(tx, id):
    result = tx.run("MATCH (f:Observation)-[rel:ADJACENT]->(t:Observation) WHERE ID(f) = $id RETURN collect(t) as next_adjacencies", id=int(id))
    r = result.single()
    return r["next_adjacencies"]

def get_observation_group(tx, id):
    result = tx.run("MATCH (o:Observation) WHERE ID(o) = $id RETURN o.group2 AS group", id=int(id))
    r = result.single()
    return r["group"]


def insert_observation(tx, t, y, x, f):
    result = tx.run("Create(o:Observation {t:$t, y:$y, x:$x, f:$f}) RETURN ID(o)", t=t, y=y, x=x, f=f)
    r = result.single()
    return r["ID(o)"]


def insert_adjacency(tx, from_id, to_id, distance):
    tx.run( "MATCH (from:Observation) WHERE ID(from) = $from_id "
            "MATCH (to:Observation) WHERE ID(to) = $to_id "
            "CREATE (from)-[rel:ADJACENT {d: $distance}]->(to)", from_id=int(from_id), to_id=int(to_id), distance=float(distance))
          


class MemoryGraph:
    def __init__(self, index_path=None, db="neo4j://localhost:7687"):
        self.index_path = index_path
        self.db = db
        self.last_id = None
        self.predictions = None
        self.open()


    def open(self):
        #knn
        self.index = hnswlib.Index(space='l2', dim=256)
        
        if self.index_path != None:
            self.index.load_index(self.index_path)
        else:
            self.init_index()
        
        self.index.set_ef(1000)   

        #db
        self.driver = GraphDatabase.driver(self.db, auth=("neo4j", "password"))
        

    def save_index(self, index_path):
        self.index_path = index_path
        self.index.save_index(index_path)

    def load_index(self):
        self.index.load_index(self.index_path)

    def init_index(self):
        self.index.init_index(max_elements=50000, ef_construction=1000, M=64)

    def close(self):
        self.driver.close()

    def knn_query(self, feats, k=1):
        return self.index.knn_query(feats, k)

    def get_observation_group(self, label):
        with self.driver.session() as session:
            return session.read_transaction(get_observation_group, label)

    # o: a feature vector
    def add_observation(self, t, pos, feats, adjacency_broken):

        # if there is a feat that is almost identical use that id instead
        l = d = None

        if self.index.get_current_count() >= knn:
            labels, distances = self.index.knn_query(feats, k = knn)
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
                    feats_to_json(feats[0])
                )

        # Save Previous -> Current Adjacency 
        if self.last_id != None:
            with self.driver.session() as session:
                session.write_transaction(
                    insert_adjacency, 
                    self.last_id,
                    oid,
                    0.0
                )

        if self.predictions != None:

            accurate_predictions = 0

            for pred in self.predictions:
                f = np.array(json.loads(pred['candidate_for_similar_to_curr']["f"]))
                distance = np.linalg.norm(feats[0]-f)
                
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

            if len(self.predictions) > 0:
                print("Predictions", accurate_predictions, "of", len(self.predictions))


        self.predictions = []

        if l is not None and d is not None:

            similar = 0

            for n in range(knn):
                label = l[n]
                distance = d[n]

                if distance <= distance_threshold:
                    # Found a previous similar observation
                    with self.driver.session() as session:
                        next_adjacencies = session.write_transaction(
                            get_next_adjacencies, 
                            label
                        )

                        for n in next_adjacencies:
                            self.predictions.append(dict(id_similar_to_prev=label, candidate_for_similar_to_curr=n))

                    similar += 1

            if similar > 0:
                print("Similar ", similar)

        self.index.add_items(feats, [oid])
        self.last_id = oid

        return oid