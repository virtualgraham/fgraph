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


# Camera Motion

The walk grid should transform with with camera motion, rotation translation scale 

# Change Detection

Use same type of change detection as video encoders for attention. Adjust for camera movement transformations

# Orientation detection
Instead of having orientation invariant feature detectors, there should be an orientation detector that determines the walk grid orientation even before tracking

# Edge Meaning
Edges should represent actions, specifically "looking in a particular direction". So if you have have seen a clearly identified feature you should be able to estimate what you see by looking right or looking left.  It should require less of an exact match for a feature that is in the place it is expected.

# Feature fusion

Multiple different kind of features should be able to added to graph

# Shape of objects

The shape of objects needs to be added as one of the new features
perhaps some edge detection features 
When edges are connected in a graph they represent different shapes?

# Predictions

Predictions should not be a simple neighborhood search. It should be a search though candidates of what objects are being observed. Try fully connecting small neighborhoods to connect larger neighborhoods.


## 

Given a video with a single sample masked object, locate that object in a set of videos with multiple objects that may or may not contain the sample object.
    The program will produce a collection of observations
        
        What percent of the produced observations are of the sample class?
        What percent of the produced observations are not of the sample class?



        [True Positive] What percent of observations in complete set of observations of sample class are in produced observations? 100%
        [False Negative] What percent of observations in complete set of observations of sample class are not in produced observations? 0%
        [False Positive] What percent of observations in complete set of observations not of sample class are in produced observations? 0%
        [True Negative] What percent of observations in complete set of observations not of sample class are not in produced observations? 100%

 

        [True Positive]  What percent of videos with sample class have produced observations of the sample class?
        (number of unique videos in produced observations labeled with sample class) / (total number of videos belonging to sample class)
        
        [False Negative] What percent of videos with sample class do not have produced observations of the sample class?
        1 - [True Positive]

        [False Positive] What percent of videos without sample class have observations of the sample class?
        (number of unique videos in produced observations NOT labeled with sample class) / (total number of videos not belonging to sample class)

        [True Negative] What percent of videos without sample class have observations not of the sample class?
        1 - [False Positive]



        # frames with sample class are frames with more than a threshold number of pixels belonging to sample class 

        [True Positive]  What percent of frames with sample class have produced observations of the sample class?
        (number of unique frames in produced observations labeled with sample class) / (total number of frames with sample class)

        [False Negative] What percent of frames with sample class do not have produced observations of the sample class?
        1 - [True Positive]

        [False Positive] What percent of frames without sample class have produced observations of the sample class?
        (number of unique frames in produced observations NOT labeled with sample class) / (total number of frames without sample class)

        [True Negative] What percent of frames without sample class  have no produced observations of the sample class?
        1 - [False Positive]



        # Where the pixels inside the observation window are all assumed to be of the same class

        [True Positive]  What percent of pixels belonging to sample class are in the unique set of pixels of produced observations? 
        (number of unique pixels in produced observations labeled with sample class) / (total number of pixels belonging to class)

        [False Negative] What percent of pixels belonging to sample class are not in the unique set of pixels of produced observations? 
        1 - [True Positive]

        [False Positive] What percent of pixels not belonging to sample class are in the unique set of pixels of produced observations? 
        (number of unique pixels in produced observations NOT labeled with sample class) / (total number of pixels not belonging to class)

        [True Negative] What percent of pixels not belonging to sample class are not in the unique set of pixels of produced observations?
        1 - [False Positive]


        For each frame there is a grid with each element being the size of the center of the window (used to determine the class of the observation)
        Each grid element is labeled with the class the majority of correctly labeled observations would be if centered anywhere in that element



    Summary Statistics needed to Calculate Above Measures Quickly
        Total number of observations
        Total number of observations of each class

        Total number of videos
        Total number of videos of each class

        Total number of frames
        for each class total number of frames belonging to class

        Total number of pixels
        for each class total number of pixels belonging to class


Given a set of videos each which contain a random subset of objects, locate each object wherever it appears.
    The program will produce multiple sets of observations, where each set presumably represents all the observations for a class of objects.
    For each set in the produced observations:
        For each known class:
            Calculate the stats for the object retrieval task.
    Ideally each set should only score well with zero or one classes and there should only be one set that scores well for each known class.

# TODO
    in build_graph
        add object to observation before inserting
        call increment_video_counts after each video
        call increment_frame_counts after each frame
