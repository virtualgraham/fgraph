import community_walk_graph as cwg

# Build a ring graph with 20 nodes
graph = cwg.new_graph()
cwg.add_edge(graph, 1, 2)
cwg.add_edge(graph, 2, 3)
cwg.add_edge(graph, 3, 4)
cwg.add_edge(graph, 4, 5)
cwg.add_edge(graph, 5, 6)
cwg.add_edge(graph, 6, 7)
cwg.add_edge(graph, 7, 8)
cwg.add_edge(graph, 8, 9)
cwg.add_edge(graph, 9, 10)
cwg.add_edge(graph, 10, 11)
cwg.add_edge(graph, 11, 12)
cwg.add_edge(graph, 12, 13)
cwg.add_edge(graph, 13, 14)
cwg.add_edge(graph, 14, 15)
cwg.add_edge(graph, 15, 16)
cwg.add_edge(graph, 16, 17)
cwg.add_edge(graph, 17, 18)
cwg.add_edge(graph, 18, 19)
cwg.add_edge(graph, 19, 20)
cwg.add_edge(graph, 20, 1)

# Query 5 nodes from each side of the ring
communities_result = cwg.communities(graph, [1,2,3,4,5, 11,12,13,14,15], 2, 1000, 0)
print("communities", communities_result)

communities_result = cwg.communities_range(graph, [1,2,3,4,5, 11,12,13,14,15], 0, 1, 1000, 0)
print("communities_range", communities_result)