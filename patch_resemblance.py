import hnswlib
import os
import math
from os import listdir, path
from database import PatchGraphDatabase
import numpy as np

# for each scene
# list all patches
# find nearest neighbors of each patch
# exlude nodes that are already closely connected via other relationship types
# add resembles relationship from scene patch to nearest descriptor neighbor

sun_rgbd_directory = "/Users/user/Desktop/SUNRGBD"

def list_image_paths(collection_dir):
    scene_dirs = [path.join(collection_dir, d, "image") for d in listdir(collection_dir) if path.isdir(path.join(collection_dir, d))]
    image_paths = [[path.join(d, e) for e in listdir(d)][0] for d in scene_dirs]
    return [(p[len(sun_rgbd_directory)+1:], p) for p in image_paths]

print("Loading desc_index.bin")

desc_index = hnswlib.Index(space='l2', dim=128)
desc_index.load_index("desc_index.bin", max_elements = 7000000)

database = PatchGraphDatabase()

images = list_image_paths(path.join(sun_rgbd_directory, "kv2", "kinect2data")) + list_image_paths(path.join(sun_rgbd_directory, "kv2", "align_kv2"))

print("Starting")

# mark = False
count = 0

for image_name, image_path in images:
    
    print(count, image_name)
    count += 1
    
    # if image_name == "kv2/kinect2data/003045_2014-06-15_13-37-25_094959634447_rgbf000061-resize/image/0000061.jpg":
    #     print("Found Mark")
    #     mark = True
    # if not mark:
    #     continue

    patches = database.list_scene_patches(image_name)
    
    patch_id_set = set()
    for patch in patches:
        patch_id_set.add(int(patch['id']))

    resemblances = []

    for patch in patches:
        
        patch_id = int(patch['id'])

        descriptor = np.array(patch['des'])
        labels, distances = desc_index.knn_query(descriptor, k=10)
        
        for i in range(labels.shape[1]):

            distance = math.sqrt(float(distances[0,i]))
            if distance > 0.75:
                break

            label = int(labels[0,i])
            
            if label in patch_id_set:
                continue

            resemplance = {'from': patch_id, 'to': label, 'dist': distance}
            resemblances.append(resemplance)

    insert_result = database.insert_resembles_relationships(resemblances)
    print(len(resemblances), len(insert_result))