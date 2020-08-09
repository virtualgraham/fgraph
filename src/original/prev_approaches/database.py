from neo4j import GraphDatabase
from neo4j.types.spatial import CartesianPoint
import json

batch_size = 100


# CREATE INDEX ON :Patch(scene)

class PatchGraphDatabase:
    def __init__(self, url='bolt://localhost:7687', user='neo4j', password='password'):
        self._driver = GraphDatabase.driver(url, auth=(user, password))

    def close(self):
        self._driver.close()

    def insert_patches(self, patches):
        with self._driver.session() as session:
            return session.write_transaction(PatchGraphDatabase._insert_patches, patches)

    def insert_scene(self, scene):
        with self._driver.session() as session:
            return session.write_transaction(PatchGraphDatabase._insert_scene, scene)

    def insert_resembles_relationships(self, relationships):
        with self._driver.session() as session:
            return session.write_transaction(PatchGraphDatabase._insert_resembles_relationships, relationships)

    def insert_scene_neighbor_relationships(self, relationships):
        with self._driver.session() as session:
            return session.write_transaction(PatchGraphDatabase._insert_scene_neighbor_relationships, relationships)

    def insert_scene_contains_relationships(self, relationships):
        with self._driver.session() as session:
            return session.write_transaction(PatchGraphDatabase._insert_scene_contains_relationships, relationships)

    def get_patchs(self, ids):
        with self._driver.session() as session:
            return session.write_transaction(PatchGraphDatabase._get_patchs, ids)

    def list_scene_patches(self, scene):
        with self._driver.session() as session:
            return session.write_transaction(PatchGraphDatabase._list_scene_patches, scene)

    def list_scene_patch_ids(self, scene):
        with self._driver.session() as session:
            return session.write_transaction(PatchGraphDatabase._list_scene_patch_ids, scene)

    def find_paths_between_scenes(self, scene1, scene2):
        with self._driver.session() as session:
            return session.write_transaction(PatchGraphDatabase._find_paths_between_scenes, scene1, scene2)

    @staticmethod
    def _cartesian_point_to_tuple_2d(point):
        return (point.x, point.y)

    @staticmethod
    def _patch_from_node(node):
        patch = {'id': node.id, 'loc': PatchGraphDatabase._cartesian_point_to_tuple_2d(node.get('loc')), 'des': json.loads(node.get('des')), 'scene': node.get('scene'), 'size': node.get('size')} # , 'angle': node.get('angle')
        return patch

    @staticmethod
    def _scene_from_node(node):
        patch = {'id': node.id, 'scene': node.get('scene')}
        return patch

    @staticmethod
    def _resembles_from_node(node):
        patch = {'id': node.id, 'dist': node.get('dist')}
        return patch

    @staticmethod
    def _scene_neighbor_from_node(node):
        patch = {'id': node.id, 'dist': node.get('dist')}
        return patch

    @staticmethod
    def _scene_contains_from_node(node):
        patch = {'id': node.id}
        return patch

    @staticmethod
    def _get_patchs(tx, ids):
        # query = "MATCH (patch:Patch) WHERE ID(patch) = $id RETURN patch"
        # result = tx.run(query, id=id)    
        # return [PatchGraphDatabase._patch_from_node(record['patch']) for record in result]

        items = [{"id": int(id)} for id in ids]
        query = "UNWIND {props} as prop MATCH (patch:Patch) WHERE ID(patch) = prop.id RETURN patch"
        record_to_object_func = PatchGraphDatabase._patch_from_node
        unwind_name = 'props'
        return_name='patch'

        return PatchGraphDatabase._batch_insert(tx, items, query, record_to_object_func, unwind_name, return_name)

    @staticmethod
    def _list_scene_patches(tx, scene):
        query = "MATCH (patch:Patch) WHERE patch.scene = $scene RETURN patch"
        result = tx.run(query, scene=scene)    
        return [PatchGraphDatabase._patch_from_node(record['patch']) for record in result]

    @staticmethod
    def _list_scene_patch_ids(tx, scene):
        query = "MATCH (patch:Patch) WHERE patch.scene = $scene RETURN ID(patch)"
        result = tx.run(query, scene=scene)    
        return [record['ID(patch)'] for record in result]

    @staticmethod
    def _find_paths_between_scenes(tx, scene1, scene2):
        query = "MATCH path = (s1:Scene)-[:CONTAINS]-(:Patch)-[:RESEMBLES]-(:Patch)-[:CONTAINS]-(s2:Scene) WHERE s1.scene = $scene1 AND s2.scene = $scene2 RETURN path"
        result = tx.run(query, scene1=scene1, scene2=scene2)    

        foo = [[path for path in record['path']] for record in result]
        return [(PatchGraphDatabase._patch_from_node(bar[0].nodes[1]), PatchGraphDatabase._patch_from_node(bar[2].nodes[1])) for bar in foo]

    @staticmethod
    def _insert_resembles_relationships(tx, relationships):
    
        items = [{"from": int(r['from']), "to": int(r['to']), "dist": float(r['dist'])} for r in relationships]
        query = "UNWIND {props} as pair MATCH (a:Patch), (b:Patch) WHERE ID(a) = pair.from AND ID(b) = pair.to CREATE (a)-[n:RESEMBLES {dist: pair.dist}]->(b) RETURN n"
        record_to_object_func = PatchGraphDatabase._resembles_from_node
        unwind_name = 'props'
        return_name='n'

        return PatchGraphDatabase._batch_insert(tx, items, query, record_to_object_func, unwind_name, return_name)

    @staticmethod
    def _insert_scene_neighbor_relationships(tx, relationships):
    
        items = [{"from": int(r['from']), "to": int(r['to']), "dist": float(r['dist'])} for r in relationships]
        query = "UNWIND {props} as pair MATCH (a:Patch), (b:Patch) WHERE ID(a) = pair.from AND ID(b) = pair.to CREATE (a)-[n:NEIGHBOURS {dist: pair.dist}]->(b) RETURN n"
        record_to_object_func = PatchGraphDatabase._scene_neighbor_from_node
        unwind_name = 'props'
        return_name='n'

        return PatchGraphDatabase._batch_insert(tx, items, query, record_to_object_func, unwind_name, return_name)

    @staticmethod
    def _insert_scene_contains_relationships(tx, relationships):

        items = [{"from": int(r['from']), "to": int(r['to'])} for r in relationships]
        query = "UNWIND {props} as pair MATCH (a:Scene), (b:Patch) WHERE ID(a) = pair.from AND ID(b) = pair.to CREATE (a)-[n:CONTAINS]->(b) RETURN n"
        record_to_object_func = PatchGraphDatabase._scene_contains_from_node
        unwind_name = 'props'
        return_name='n'
            
        return PatchGraphDatabase._batch_insert(tx, items, query, record_to_object_func, unwind_name, return_name)
        
    @staticmethod
    def _insert_patches(tx, patches):
        
        items = [{"des": json.dumps(p['des'].tolist(), separators=(',', ':')), "scene": p['scene'], "size": p['size'], "loc": CartesianPoint(( float(p['loc'][0]), float(p['loc'][1] ))) }  for p in patches] #, "angle": p['angle']
        query = "UNWIND {props} AS properties CREATE (patch:Patch) SET patch = properties RETURN patch"
        record_to_object_func = PatchGraphDatabase._patch_from_node
        unwind_name = 'props'
        return_name='patch'
            
        return PatchGraphDatabase._batch_insert(tx, items, query, record_to_object_func, unwind_name, return_name)

    @staticmethod
    def _batch_insert(tx, items, query, record_to_object_func, unwind_name='props', return_name='n'):

        count = len(items)
        batches = -(-count // batch_size)

        inserted = []

        for j in range(0, batches):

            props = []

            for k in range(0, batch_size):

                i = j * batch_size + k
                if i >= count: 
                    break

                props.append(items[i])

            result = tx.run(query, **{unwind_name: props})            
            inserted.extend([(record_to_object_func(record[return_name])) for record in result])

        return inserted


    # @staticmethod
    # def _insert_patches(tx, patches):
        
    #     # des: 128 dim patch descriptor
    #     # scene: image name
    #     # loc: coordinates of patch in scene

    #     points = len(patches)
    #     batches = -(-points // batch_size)

    #     inserted_patches = []

    #     for j in range(0, batches):

    #         props = []
            
    #         query = "UNWIND $props AS properties CREATE (patch:Patch) SET patch = properties RETURN patch"

    #         for k in range(0, batch_size):

    #             i = j * batch_size + k
    #             if i >= points: 
    #                 break

    #             data = {"des": json.dumps(patches[i]['des'].tolist(), separators=(',', ':')), "scene": patches[i]['scene'], "size": patches[i]['size'], "loc": CartesianPoint(( float(patches[i]['loc'][0]), float(patches[i]['loc'][1] ))) } 
    #             props.append(data)

    #         result = tx.run(query, props=props)   
    #         #print("inserted nodes", result.summary().counters.nodes_created)          
    #         inserted_patches.extend([(PatchGraphDatabase._patch_from_node(record['patch'])) for record in result])

    #     return inserted_patches

    # @staticmethod
    # def _insert_scene_neighbor_relationships(tx, relationships):

    #     count = len(relationships)
    #     batches = -(-count // batch_size)

    #     inserted = []

    #     for j in range(0, batches):

    #         props = []
            
    #         query = "UNWIND {props} as pair MATCH (a:Patch), (b:Patch) WHERE ID(a) = pair.from AND ID(b) = pair.to CREATE (a)-[n:NEIGHBOURS {dist: pair.dist}]->(b) RETURN n"

    #         for k in range(0, batch_size):

    #             i = j * batch_size + k
    #             if i >= count: 
    #                 break

    #             data = { "from": relationships[i]['from'], "to": relationships[i]['to'], "dist": relationships[i]['dist'] } 
    #             props.append(data)

    #         result = tx.run(query, props=props)   
    #         #print("inserted relationships", result.summary().counters.relationships_created)          
    #         inserted.extend([(PatchGraphDatabase._scene_neighbor_from_node(record['n'])) for record in result])

    #     return inserted

    # @staticmethod
    # def _insert_scene_contains_relationships(tx, relationships):

    #     count = len(relationships)
    #     batches = -(-count // batch_size)

    #     inserted = []

    #     for j in range(0, batches):

    #         props = []
            
    #         query = "UNWIND {props} as pair MATCH (a:Scene), (b:Patch) WHERE ID(a) = pair.from AND ID(b) = pair.to CREATE (a)-[n:CONTAINS]->(b) RETURN n"

    #         for k in range(0, batch_size):

    #             i = j * batch_size + k
    #             if i >= count: 
    #                 break

    #             data = { "from": relationships[i]['from'], "to": relationships[i]['to'] } 
    #             props.append(data)

    #         result = tx.run(query, props=props)  
    #         #print("inserted relationships", result.summary().counters.relationships_created)               
    #         inserted.extend([(PatchGraphDatabase._scene_contains_from_node(record['n'])) for record in result])

    #     return inserted

    @staticmethod
    def _insert_scene(tx, scene):

        query = "CREATE (s:Scene {scene:$scene}) RETURN s"

        result = tx.run(query, scene=scene['scene'])      
        #print("inserted nodes", result.summary().counters.nodes_created)  
        return [(PatchGraphDatabase._scene_from_node(record['s'])) for record in result]