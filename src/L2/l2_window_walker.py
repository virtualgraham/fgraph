from L2_Net import L2Net

import cv2
import math
import random
import numpy as np
import json 
import hnswlib

from neo4j import GraphDatabase


colors = [
    (1, 0, 103),
    (213, 255, 0),
    (255, 0, 86),
    (158, 0, 142),
    (14, 76, 161),
    (255, 229, 2),
    (0, 95, 57),
    (0, 255, 0),
    (149, 0, 58),
    (255, 147, 126),
    (164, 36, 0),
    (0, 21, 68),
    (145, 208, 203),
    (98, 14, 0),
    (107, 104, 130),
    (0, 0, 255),
    (0, 125, 181),
    (106, 130, 108),
    (0, 174, 126),
    (194, 140, 159),
    (190, 153, 112),
    (0, 143, 156),
    (95, 173, 78),
    (255, 0, 0),
    (255, 0, 246),
    (255, 2, 157),
    (104, 61, 59),
    (255, 116, 163),
    (150, 138, 232),
    (152, 255, 82),
    (167, 87, 64),
    (1, 255, 254),
    (255, 238, 232),
    (254, 137, 0),
    (189, 198, 255),
    (1, 208, 255),
    (187, 136, 0),
    (117, 68, 177),
    (165, 255, 210),
    (255, 166, 254),
    (119, 77, 0),
    (122, 71, 130),
    (38, 52, 0),
    (0, 71, 84),
    (67, 0, 44),
    (181, 0, 255),
    (255, 177, 103),
    (255, 219, 102),
    (144, 251, 146),
    (126, 45, 210),
    (189, 211, 147),
    (229, 111, 254),
    (222, 255, 116),
    (0, 255, 120),
    (0, 155, 255),
    (0, 100, 1),
    (0, 118, 255),
    (133, 169, 0),
    (0, 185, 23),
    (120, 130, 49),
    (0, 255, 198),
    (255, 110, 65),
    (232, 94, 190),
    (0, 0, 0),
]

# randomly walk window over frames of video
# calculate CNN features for each window

def move(pos):
    m = random.randint(0, 3)

    if m == 0: # up
        if pos[0] == 0:
            return move(pos)
        #print("up")
        return (pos[0]-1, pos[1])
    elif m == 1: # right
        if pos[1] == steps[1]-1:
            return move(pos)
        #print("right")
        return (pos[0], pos[1]+1)
    elif m == 2: # down
        if pos[0] == steps[0]-1:
            return move(pos)
        #print("down")
        return (pos[0]+1, pos[1])
    else: # left
        if pos[1] == 0:
            return move(pos)
        #print("left")
        return (pos[0], pos[1]-1)


def resize_frame(image, window_size = 224, stride=32, steps=(15,15)):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    target_height = window_size + (steps[0]-1) * stride
    target_width = window_size + (steps[1]-1) * stride 
    
    target_wh_ratio = target_width/target_height
    
    image_height = image.shape[0]
    image_width = image.shape[1]
    
    image_wh_ratio = image_width/image_height
    
    # if image is taller and skinner than target, scale width first then crop height
    # else if image is shorter and fatter than target, scale height first then crop width
    
    
    if image_wh_ratio < target_wh_ratio:
        scale_percent = target_width/image.shape[1]
        scale_height = math.floor(scale_percent * image.shape[0])
        image = cv2.resize(image, (target_width, scale_height), interpolation=cv2.INTER_CUBIC)
        m1 = (scale_height - target_height)//2
        m2 = target_height + m1
        frame = image[m1:m2,:]
    
    else:
        scale_percent = target_height/image.shape[0]
        scale_width = math.floor(scale_percent * image.shape[1])
        image = cv2.resize(image, (scale_width, target_height), interpolation=cv2.INTER_CUBIC)
        m1 = (scale_width-target_width)//2
        m2 = target_width + m1
        frame = image[:,m1:m2]

    return np.reshape(frame, (frame.shape[0], frame.shape[1], 1))


def feats_to_json(a):
    return json.dumps(a.tolist())


def json_to_feats(s):
    load = json.loads(s)
    return np.asarray(load)


def get_observation_group(tx, id):
    result = tx.run("MATCH (o:Observation) WHERE o.id = $id RETURN o.group11 AS group", id=int(id))
    r = result.single()
    print(r)
    return r["group"]


def insert_observation(tx, id, t, y, x, f):
    tx.run("Create(o:Observation {id:$id, t:$t, y:$y, x:$x, f:$f})", id=id, t=t, y=y, x=x, f=f)


def insert_adjacency(tx, from_id, to_id, distance):
    tx.run( "MATCH (from:Observation {id: $from_id}) "
            "MATCH (to:Observation {id: $to_id}) "
            "CREATE (from)-[rel:ADJACENT {d: $distance}]->(to)", from_id=int(from_id), to_id=int(to_id), distance=float(distance))
           
def window_id(t, y, x):
    return t*steps[0]*steps[1] + y*steps[0] + x


runs = 10
knn = 10
window_size = 64
stride=32
steps=(33,58) # height width
frame_batch_size=25
max_batches=100

distance_threshold = 1.5

# for each frame
# for each window
# cnn feats
# nearest neighbors
# db query for the nearest neighbor
# draw translucent color rectangle based on group

def play_annotated_video():

    # Video
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture('./media/cows.mp4')

    # CNN
    l2_net = L2Net("L2Net-HP+", True)

    # KNN
    p = hnswlib.Index(space='l2', dim=256)
    p.load_index("./data/index.bin")

    # DB
    uri = "neo4j://localhost:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))


    group_id = 0
    group_id_dict = {}

    while vc.isOpened():

        rval, frame = vc.read()
        if rval == False:
            break
                
        frame = resize_frame(frame, window_size, stride, steps)

        # print(frame.shape)

        windows = np.empty((steps[0]*steps[1], window_size, window_size, 1))

        for x in range(steps[0]):
            for y in range(steps[1]):
                w = x * steps[0] + y
                #print((stride*y), (stride*y+window_size), (stride*x), (stride*x+window_size))
                windows[w] = frame[(stride*x):(stride*x+window_size), (stride*y):(stride*y+window_size)]
                #cv2.imwrite('./output/testing'+str(w)+'.jpg', windows[w])


        # extract cnn features from windows
        feats = l2_net.calc_descriptors(windows)

        labels, distances = p.knn_query(feats, k = 1)

        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        for x in range(steps[0]):
            for y in range(steps[1]):
                label = labels[x * steps[0] + y][0]

                # get category of observation with label
                with driver.session() as session:
                    g = session.read_transaction(get_observation_group, label)
                    if g not in group_id_dict:
                        group_id_dict[g] = group_id
                        group_id += 1
                    g = group_id_dict[g]

                print("observation_group", g)

                if g >= len(colors):
                    c = colors[len(colors)-1]
                else:
                    c = colors[g]

                cv2.circle(frame, (stride*y+round(window_size/2), stride*x+round(window_size/2)), 3, c, cv2.FILLED)

        cv2.imshow("preview", frame)

        key = cv2.waitKey(0)

        if key == 27: # exit on ESC
            break

    vc.release()
    cv2.destroyWindow("preview")   




def build_graph():

    print("Starting...")

    #initialize CNN and KNN index
    l2_net = L2Net("L2Net-HP+", True)

    #initialize KNN index
    p = hnswlib.Index(space='l2', dim=256)
    p.init_index(max_elements=50000, ef_construction=100, M=16)
    p.set_ef(10)

    #initialize graph database
    uri = "neo4j://localhost:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))


    total_frame_count = 0

    # for each run though the video
    for r in range(runs):

        print("Run", r)

        # open video file for a run though
        cap = cv2.VideoCapture('./media/cows.mp4')
    
        # select a random starting position
        pos = (random.randint(0, steps[0]-1),random.randint(0, steps[1]-1))

        done = False

        last_label = None
        last_labels = None
        last_distances = None
        
        run_frame_count = 0

        # for each batch
        for t in range(max_batches):
            if done:
                break

            print("Batch", t)

            windows = np.empty((frame_batch_size, window_size, window_size, 1))
            positions = []
            ids = []
            batch_frame_count = 0

            # read frames from video and walk window
            for b in range(frame_batch_size):
                ret, frame = cap.read()
                
                if ret == False:
                    done = True
                    break
                
                print("pos", pos)

                print("frame.shape", frame.shape)
                frame = resize_frame(frame, window_size, stride, steps)
                print("frame.shape", frame.shape)


                windows[b] = frame[(stride*pos[0]):(stride*pos[0]+window_size), (stride*pos[1]):(stride*pos[1]+window_size)]

                cv2.imwrite('./output/testing'+str(total_frame_count)+'.jpg',windows[b])

                positions.append(pos)
                
                t = run_frame_count-batch_frame_count+b
                ids.append(window_id(t,pos[0],pos[1]))

                total_frame_count+=1
                batch_frame_count+=1
                run_frame_count+=1

                pos = move(pos)

            # if no frames were read break
            if batch_frame_count == 0:
                break

            # if batch is short resize windows array to match
            if batch_frame_count != frame_batch_size:
                windows = windows[0:batch_frame_count]

            # extract cnn features from windows
            feats = l2_net.calc_descriptors(windows)
            print("feats.shape", feats.shape)

            for b in range(batch_frame_count):

                id = ids[b]

                t = run_frame_count-batch_frame_count+b
                y = positions[b][0]
                x = positions[b][1]
                
                # print(t,y,x,id)

                with driver.session() as session:
                    session.write_transaction(
                        insert_observation, 
                        id,
                        t, 
                        y, 
                        x, 
                        feats_to_json(feats[b])
                    )

            if p.get_current_count() >= knn:

                labels, distances = p.knn_query(feats, k = knn)

                for b in range(batch_frame_count):

                    current_label = ids[b]

                    if b == 0:
                        if last_labels is None or last_distances is None:
                            last_label = current_label
                            continue
                        l = last_labels[last_labels.shape[0]-1]
                        d = last_distances[last_labels.shape[0]-1]
                    else:
                        l = labels[b-1]
                        d = distances[b-1]

                    print("--", last_label, current_label)

                    with driver.session() as session:
                        session.write_transaction(
                            insert_adjacency, 
                            last_label,
                            current_label,
                            0.0
                        )

                    for n in range(knn):
                        label = l[n]
                        distance = d[n]

                        
                        if distance <= distance_threshold:

                            print("distance", distance)

                            with driver.session() as session:
                                session.write_transaction(
                                    insert_adjacency, 
                                    label,
                                    current_label,
                                    distance
                                )

                    last_label = current_label

                last_labels = labels
                last_distances = distances

            p.add_items(feats, ids)


        cap.release()
        cv2.destroyAllWindows()

    p.save_index("./data/index.bin")

    driver.close()
    print("Done")



#build_graph()
play_annotated_video()

