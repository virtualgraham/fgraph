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




# Using History and Cluster Likelihood

With history length h maybe around 7

For each observation in history
    For each nearest neighbor find community 
        add community to the set of communities


For each community
    For each history
        Does Community share nodes with nearest neighbors of history?

Sort communities by most unique history with near nodes and how close those similarities are



video_objects ['apple', 'bear', 'carrot', 'chain', 'clippers', 'cup', 'notebook', 'opener']

counts 

{
'observation_count': 90800, 
'observation_objects': {
    'c:o:apple': 1439, 
    'c:o:bear': 6678, 
    'c:o:carrot': 956, 
    'c:o:chain': 2263, 
    'c:o:clippers': 789, 
    'c:o:cup': 640, 
    'c:o:notebook': 431, 
    'c:o:opener': 425
    }, 

'frame_count': 454, 
'frame_objects': {
    'c:f:apple': 420, 
    'c:f:bear': 454, 
    'c:f:carrot': 454, 
    'c:f:chain': 454, 
    'c:f:clippers': 454, 
    'c:f:cologne': 196, 
    'c:f:cup': 454, 
    'c:f:notebook': 430, 
    'c:f:opener': 436, 
    'c:f:shorts': 200
    }, 

'video_count': 1, 
'video_objects': {
    'c:v:apple': 1, 
    'c:v:bear': 1, 
    'c:v:carrot': 1, 
    'c:v:chain': 1, 
    'c:v:clippers': 1, 
    'c:v:cup': 1, 
    'c:v:notebook': 1, 
    'c:v:opener': 1
    }, 

'pixel_count': 418406400, 
'pixel_objects': {
    'c:p:apple': 9715039, 
    'c:p:bear': 43359314, 
    'c:p:carrot': 8215097, 
    'c:p:chain': 7592207, 
    'c:p:clippers': 6154470, 
    'c:p:cologne': 5493, 
    'c:p:cup': 17976043, 
    'c:p:notebook': 13790752, 
    'c:p:opener': 4502877, 
    'c:p:shorts': 413}
}



2020-08-31 19:30:30.033350
chain

len(observations) 27015

observations_of_sample_class 0.45530260966129926
observations_not_of_sample_class 0.5446973903387007

observations_true_positive 0.7628380054577028
observations_false_negative 0.23716199454229725
observations_false_positive 0.01451246405169455
observations_true_negative 0.9854875359483054

video_true_positive 1.0
video_false_negative 0.0
video_false_positive 1.2
video_true_negative -0.19999999999999996

frame_true_positive 0.7912254160363086
frame_false_negative 0.20877458396369142
frame_false_positive 0.7083000798084597
frame_true_negative 0.29169992019154034

pixel_true_positive 0.18729040716074855
pixel_false_negative 0.8127095928392515
pixel_false_positive 0.002771298406608047
pixel_true_negative 0.9972287015933919


2020-08-31 21:31:35.903478
apple

len(observations) 35585

observations_of_sample_class 0.1859772376001124
observations_not_of_sample_class 0.8140227623998876

observations_true_positive 0.807368549469318
observations_false_negative 0.19263145053068198
observations_false_positive 0.02834668939594846
observations_true_negative 0.9716533106040516

video_true_positive 1.2
video_false_negative -0.19999999999999996
video_false_positive 1.0
video_true_negative 0.0

frame_true_positive 0.7195604395604396
frame_false_negative 0.2804395604395604
frame_false_positive 0.9547826086956521
frame_true_negative 0.045217391304347876

pixel_true_positive 0.1323916210630321
pixel_false_negative 0.867608378936968
pixel_false_positive 0.005506206306518058
pixel_true_negative 0.9944937936934819



2020-08-31 22:13:10.788614
apple

len(observations) 9968

observations_of_sample_class 0.4623796147672552
observations_not_of_sample_class 0.5376203852327448

observations_true_positive 0.5622788825179944
observations_false_negative 0.4377211174820056
observations_false_positive 0.005244240289739628
observations_true_negative 0.9947557597102604

video_true_positive 1.2
video_false_negative -0.19999999999999996
video_false_positive 1.0
video_true_negative 0.0

frame_true_positive 0.6356043956043956
frame_false_negative 0.36439560439560437
frame_false_positive 0.4761739130434783
frame_true_negative 0.5238260869565217

pixel_true_positive 0.09596063561198184
pixel_false_negative 0.9040393643880181
pixel_false_positive 0.001048156454263501
pixel_true_negative 0.9989518435457365


2020-09-01 02:32:40.966864
apple
len(observations) 10994

observations_of_sample_class 0.4741677278515554
observations_not_of_sample_class 0.5258322721484445

observations_true_positive 0.5805769016594275 5213 8979
observations_false_negative 0.41942309834057245
observations_false_positive 0.005355649000714272 5781 1079421
observations_true_negative 0.9946443509992857

video_true_positive 1.0 6 6
video_false_negative 0.0
video_false_positive 1.0 6 6
video_true_negative 0.0

frame_true_positive 0.6182458888018794 1579 2554
frame_false_negative 0.38175411119812064
frame_false_positive 0.5090027700831025 1470 2888
frame_true_negative 0.49099722991689754

pixel_true_positive 0.10141758383625014 4648285 45833127
pixel_false_negative 0.8985824161637499
pixel_false_positive 0.0010874161780444968 5403930 4969514073
pixel_true_negative 0.9989125838219555


2020-09-01 12:48:37.041876
apple
len(observations) 9527

observations_of_sample_class 0.4697176445890627
observations_not_of_sample_class 0.5302823554109373

observations_true_positive 0.5459314383310967 4475 8197
observations_false_negative 0.4540685616689033
observations_false_positive 0.00494381450714025 5052 1021883
observations_true_negative 0.9950561854928598

video_true_positive 1.2 6 5
video_false_negative -0.19999999999999996
video_false_positive 1.0 6 6
video_true_negative 0.0

frame_true_positive 0.625934065934066 1424 2275
frame_false_negative 0.374065934065934
frame_false_positive 0.4657391304347826 1339 2875
frame_true_negative 0.5342608695652173

pixel_true_positive 0.09311959976477088 3986229 42807626
pixel_false_negative 0.9068804002352291
pixel_false_positive 0.0010119212144539275 4759503 4703432374
pixel_true_negative 0.9989880787855461


2020-09-01 05:47:06.692432
apple
len(observations) 16802

observations_of_sample_class 0.32269967860968934
observations_not_of_sample_class 0.6773003213903106

observations_true_positive 0.6038534357946319 5422 8979
observations_false_negative 0.3961465642053681
observations_false_positive 0.010542689089799068 11380 1079421
observations_true_negative 0.989457310910201

video_true_positive 1.0 6 6
video_false_negative 0.0
video_false_positive 1.0 6 6
video_true_negative 0.0

frame_true_positive 0.6503523884103367 1661 2554
frame_false_negative 0.3496476115896633
frame_false_positive 0.7236842105263158 2090 2888
frame_true_negative 0.2763157894736842

pixel_true_positive 0.1035045459586469 4743937 45833127
pixel_false_negative 0.8964954540413531
pixel_false_positive 0.002122281342818123 10546707 4969514073
pixel_true_negative 0.9978777186571819



2020-09-01 05:38:57.608583
apple
len(observations) 10549

observations_of_sample_class 0.5078206465067779
observations_not_of_sample_class 0.49217935349322206

observations_true_positive 0.5966143223076067 5357 8979
observations_false_negative 0.4033856776923933
observations_false_positive 0.004809986094396904 5192 1079421
observations_true_negative 0.9951900139056031

video_true_positive 1.0 6 6
video_false_negative 0.0
video_false_positive 1.0 6 6
video_true_negative 0.0

frame_true_positive 0.6601409553641346 1686 2554
frame_false_negative 0.33985904463586536
frame_false_positive 0.4650277008310249 1343 2888
frame_true_negative 0.5349722991689752

pixel_true_positive 0.10187046587504274 4669042 45833127
pixel_false_negative 0.8981295341249572
pixel_false_positive 0.0009738249915204536 4839437 4969514073
pixel_true_negative 0.9990261750084796


2020-09-01 05:30:29.342974
apple
len(observations) 9965

observations_of_sample_class 0.5010536879076769
observations_not_of_sample_class 0.4989463120923231

observations_true_positive 0.5560752867802651 4993 8979
observations_false_negative 0.4439247132197349
observations_false_positive 0.0046061731242953395 4972 1079421
observations_true_negative 0.9953938268757047

video_true_positive 1.0 6 6
video_false_negative 0.0
video_false_positive 1.0 6 6
video_true_negative 0.0

frame_true_positive 0.6225528582615505 1590 2554
frame_false_negative 0.3774471417384495
frame_false_positive 0.4809556786703601 1389 2888
frame_true_negative 0.5190443213296398

pixel_true_positive 0.09801816489632051 4492479 45833127
pixel_false_negative 0.9019818351036795
pixel_false_positive 0.0009494616436716548 4718363 4969514073
pixel_true_negative 0.9990505383563284




2020-09-07 11:41:07.068343
apple
len(observations) 366

observations_of_sample_class 0.9754098360655737
observations_not_of_sample_class 0.024590163934426257

observations_true_positive 0.23564356435643563 357 1515
observations_false_negative 0.7643564356435644
observations_false_positive 0.00010080080640645126 9 89285
observations_true_negative 0.9998991991935936

video_true_positive 1.0 1 1
video_false_negative 0.0
video_false_positive None 0 0
video_true_negative None

frame_true_positive 0.4976190476190476 209 420
frame_false_negative 0.5023809523809524
frame_false_positive 0.11764705882352941 4 34
frame_true_negative 0.8823529411764706

pixel_true_positive 0.03589476069010119 348719 9715039
pixel_false_negative 0.9641052393098988
pixel_false_positive 2.255002400209776e-05 9216 408691361
pixel_true_negative 0.9999774499759979


2020-09-07 11:47:02.216136
apple
len(observations) 540

observations_of_sample_class 0.9851851851851852
observations_not_of_sample_class 0.014814814814814836

observations_true_positive 0.36264485344239944 532 1467
observations_false_negative 0.6373551465576006
observations_false_positive 8.955257295736178e-05 8 89333
observations_true_negative 0.9999104474270426

video_true_positive 1.0 1 1
video_false_negative 0.0
video_false_positive None 0 0
video_true_negative None

frame_true_positive 0.5952380952380952 250 420
frame_false_negative 0.40476190476190477
frame_false_positive 0.029411764705882353 1 34
frame_true_negative 0.9705882352941176

pixel_true_positive 0.052609258696748415 511101 9715039
pixel_false_negative 0.9473907413032516
pixel_false_positive 1.9129349788237877e-05 7818 408691361
pixel_true_negative 0.9999808706502118

8 0.19735623628447774 7
16 0.19019594007161322 9
32 0.13152029855202263 8
64 0.18929295819198166 12
128 0.2062389040208058 14
256 0.1839746732217885 24
512 0.16698854389367235 35
1024 0.12679274414130393 51
2048 0.2913655887397555 93
4096 0.4326073029343598 253