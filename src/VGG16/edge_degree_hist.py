from vgg16_window_walker_lib_d import MemoryGraph

db_path = "../../data/table_objects_mask.db"
max_elements=1200000

memory_graph = MemoryGraph(db_path, space='cosine', dim=512, max_elements=max_elements)

hist = dict()
for node in memory_graph.graph.nodes:
    d = memory_graph.graph.degree[node]
    if d in hist:
        hist[d] += 1
    else:
        hist[d] = 1
    
print(hist)