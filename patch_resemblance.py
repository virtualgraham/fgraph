import hnswlib
import math
from os import listdir, path
from database import PatchGraphDatabase
import numpy as np

# for each scene
# list all patches
# find nearest neighbors of each patch
# exlude nodes that are already closely connected via other relationship types
# add resembles relationship from scene patch to nearest descriptor neighbor

images_directory = "/Users/user/Desktop/household_images"

print("Loading desc_index.bin")

descriptor_size = 1280
desc_index = hnswlib.Index(space='cosine', dim=descriptor_size)
desc_index.load_index("desc_index.bin", max_elements=7000000)


database = PatchGraphDatabase()

images = [(e, path.join(images_directory, e)) for e in listdir(images_directory)]

print("Starting")

# mark = False

neighbor_count = 30
neighbor_distance = 0.07

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

    skipped_because_of_dup_other = 0
    skipped_because_not_similar = 0

    for patch in patches:
        
        patch_id = int(patch['id'])

        descriptor = np.array(patch['des'])
        labels, distances = desc_index.knn_query(descriptor, k=neighbor_count)
        
        # list all the patches for the labels and create a dictionary of ids to scene
        scene_dict = {}

        similar_patch_ids = [int(label) for label in labels[0]]
        similar_patches = database.get_patchs(similar_patch_ids)
        # print('len(similar_patches)', len(similar_patches))

        for similar_patch in similar_patches:
            scene_dict[similar_patch['id']] = similar_patch['scene']

        similar_scenes = set()

        for i in range(labels.shape[1]):

            distance = float(distances[0, i]) #math.sqrt(float(distances[0, i]))

            if distance > neighbor_distance:
                skipped_because_not_similar += (neighbor_count + i)
                break

            label = int(labels[0, i])

            # if resemblance is to this scene or antother scene this patch has already resembled, skip
            if label in patch_id_set: 
                continue

            similar_scene = scene_dict[label]

            if similar_scene in similar_scenes:
                skipped_because_of_dup_other += 1
                continue
                      
            similar_scenes.add(similar_scene)

            resemplance = {'from': patch_id, 'to': label, 'dist': distance}

            resemblances.append(resemplance)

    insert_result = database.insert_resembles_relationships(resemblances)

    print('skipped_because_of_dup_other', skipped_because_of_dup_other)
    print('skipped_because_not_similar', skipped_because_not_similar)
    print('len(resemblances)', len(resemblances))
