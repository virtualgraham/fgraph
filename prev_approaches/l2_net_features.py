import numpy as np
import cv2
import hnswlib
import math
from os import listdir, path
from prev_approaches.l2_net import L2Net
from database import PatchGraphDatabase

sun_rgbd_directory = "/Users/user/Desktop/SUNRGBD"

def list_image_paths(collection_dir):
    scene_dirs = [path.join(collection_dir, d, "image") for d in listdir(collection_dir) if path.isdir(path.join(collection_dir, d))]
    image_paths = [[path.join(d, e) for e in listdir(d)][0] for d in scene_dirs]
    return [(p[len(sun_rgbd_directory)+1:], p) for p in image_paths]

def open_and_prepare_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def iterate_patches(source, size):
    
    # size should be an even integer
    half_size = size // 2

    # horizontal patches count
    h = source.shape[0] // size
    # vertical patches count
    v = source.shape[1] // size  

    # offset of first horizontal patch
    oh = (source.shape[0] - (h * size))//2
    # offset of first vertical patch
    ov = (source.shape[1] - (v * size))//2

    for x in range(0, h*2-1):
        for y in range(0, v*2-1):
            yield source[ oh+x*half_size : oh+(x+2)*half_size, ov+y*half_size : ov+(y+2)*half_size ], ( oh+(x+1)*half_size, ov+(y+1)*half_size )

# generate list of all image paths
# for each path
# load image convert to greyscale
# for each size patch extract descriptors and locations
# add all patches to database
# add all descriptors to nn index

# print([b for a, b in iterate_patches(image, 256)])

_l2_net = L2Net("L2Net-HP+", False)
_database = PatchGraphDatabase()

_desc_index = hnswlib.Index(space = 'l2', dim = 128) 
_desc_index.init_index(max_elements = 7000000, ef_construction = 200, M = 16)
_desc_index.set_ef(50)

def iterate_patch_descriptors(image, image_name, size):
    for patch, location in iterate_patches(image, size):
        
        if size != 32:
            patch = cv2.resize(patch, (32,32), interpolation = cv2.INTER_CUBIC)
        
        descriptors = _l2_net.calc_descriptors(np.reshape(patch, (1, 32, 32, 1)))
        # print('descriptors.shape', descriptors.shape)

        yield {'scene': image_name, 'size': size, 'loc': location, 'des': descriptors[0]}


def extract_patches(image, image_name, scene_node, size):
    
    # print('extract_patches', image_name, size)

    patches = [p for p in iterate_patch_descriptors(image, image_name, size)]

    patches_result = _database.insert_patches(patches)
    print("inserted patch nodes", len(patches_result))

    data = np.array([p['des'] for p in patches_result], dtype=np.float32)
    data_labels = np.array([p['id'] for p in patches_result])

    _desc_index.add_items(data, data_labels)

    # insert neighbor relationships

    locations = np.array([p['loc'] for p in patches_result], dtype=np.float32)

    loc_index = hnswlib.Index(space = 'l2', dim = 2)
    loc_index.init_index(max_elements = len(patches_result), ef_construction = 1000, M = 50)
    loc_index.set_ef(1000)
    loc_index.add_items(locations, data_labels)

    relationships = []

    for i in range(0, len(patches_result)):
        from_label = data_labels[i]
        labels, distances = loc_index.knn_query(locations[i], k=5)
        for j in range(0, distances.shape[1]):
            if distances[0][j] > 0 and distances[0][j] <= (size/2)**2:
                relationships.append({"from": from_label, "to": labels[0][j], "dist": math.sqrt(distances[0][j])})

    result = _database.insert_scene_neighbor_relationships(relationships)
    print("inserted neighbor relationships", len(result))

    # insert contains relationships

    scene_contains_relationships = [{"from": scene_node['id'], "to": label} for label in data_labels]
    result = _database.insert_scene_contains_relationships(scene_contains_relationships) 
    print("inserted contains relationships", len(result))

images = list_image_paths(path.join(sun_rgbd_directory, "kv2", "kinect2data")) + list_image_paths(path.join(sun_rgbd_directory, "kv2", "align_kv2"))

print('starting', len(images))

count = 0

for image_name, image_path in images:
    
    print('images processed', count)
    image = open_and_prepare_image(image_path)

    # insert scene node
    insert_scene_result = _database.insert_scene({"scene": image_name})
    scene_node = insert_scene_result[0]
    print("inserted scene node")

    extract_patches(image, image_name, scene_node, 256)
    # extract_patches(image, image_name, scene_node, 128)
    # extract_patches(image, image_name, scene_node, 64)
    # extract_patches(image, image_name, scene_node, 32)
    # # extract_patches(image, image_name, scene_node, 16)

    count += 1
    break # FOR TESTING ONLY
    
print('saving desc_index')

_desc_index.save_index("desc_index.bin")

print('finished')