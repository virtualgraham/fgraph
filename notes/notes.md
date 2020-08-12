Patches are being saved in wrong "cafe" format

# Improve Querying 
Max pooling on query random walk
Find nearest neighbors
Find communities of the nearest neighbors
Then max pooling of the communities
Finally compare community to query

# Improve predictions
--

# Septate Feature Graph from Index of Locations
x,y,t,file data should be stored in a separate index. This way observations that are not added to the feature graph can still be referenced later. 
    - xyt entry will point to the node that was made for that location if one was made. 
    - if the observation at xyt was not made into a node because it was identical then the xyt should point to the node of the identical observation
    - if the observation was not added because it was too predictable the xyt should point to [the group of nodes that the prediction belonged to]?

node - represents a location in feature space
    has features from an observation, or combined from multiple observations

observation - represents an image patch at a location and time in a video file

node 
    primary observation
    observations that were ignored 
        nearly identical
        this node predicted it

```
# observation database schema

# insert and get observation
# obs:[observation_id] -> [observation_data]
get_observation(observation_id)
put_observation(observation)

# observations that are integrated into node's features
# node_integrated:[node_id]:[observation_id]
get_integrated_observations(node_id)
add_integrated_observation(observation_id, node_id)

# observations that were predicted by node
# node_predicted:[node_id]:[observation_id]
get_predicted_observations(node_id)
add_predicted_observation(observation_id, node_id) 

# nodes that predicted observation
# observation_predicted:[observation_id]:[node_id]
get_predicted_nodes(observation_id)

# nodes that integrate observation
# observation_integrated:[observation_id]:[node_id]
get_integrated_nodes(observation_id)

```

# New interpretation of nodes
Nodes represent areas of of feature space (spheres?) connected in a graph. 
Two new properties of nodes emerge under the new interpretation. First a integrated feature count, which is useful when tracking the mean. Also a radius value.


# need a higher graph of groups in the lower graph
perhaps each node in the lower graph is in the higher graph but has the max features of its random walked community. the connections would then connect to any node that shares a community node. This would be a much more highly connected graph. This would cause communities to become fully connected and adjacent communities to become more connected.

Maximal cliques can be merged?
https://en.wikipedia.org/wiki/Clique_problem#Listing_all_maximal_cliques