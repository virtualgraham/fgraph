from keras.applications import mobilenetv2
import numpy as np
import cv2
import hnswlib
import math
from os import listdir, path
from database import PatchGraphDatabase
import time

patch_size = 96
window_size = 448
channel_count = 3

images_directory = "/Users/user/Desktop/household_images"

feature_net = mobilenetv2.MobileNetV2(weights="imagenet", include_top=False, input_shape=(96, 96, 3))

database = PatchGraphDatabase()

desc_index = hnswlib.Index(space='cosine', dim=1280)
desc_index.init_index(max_elements=7000000, ef_construction=500, M=32)
desc_index.set_ef(500)


images = [(e, path.join(images_directory, e)) for e in listdir(images_directory)]

def open_and_prepare_image(image_path):
    image = cv2.imread(image_path)
    return image


def extract_windows(image, window_size=96, interior_w=False, interior_h=False):
    n_cells = (image.shape[0] // window_size, image.shape[1] // window_size)

    if interior_w:
        n_cells = (n_cells[0] - 1, n_cells[1])

    if interior_h:
        n_cells = (n_cells[0], n_cells[1] - 1)

    img_shape = (n_cells[0] * window_size, n_cells[1] * window_size)

    margins = ((image.shape[0] - img_shape[0])//2, (image.shape[1] - img_shape[1])//2)

    windows = np.zeros((n_cells[0] * n_cells[1], window_size, window_size, channel_count))
    coords = np.zeros((n_cells[0] * n_cells[1], 2))

    for i in range(n_cells[0]):
        for j in range(n_cells[1]):
            img = image[(margins[0] + window_size*i):(margins[0] + window_size*(i+1)), (margins[1] + window_size*j):(margins[1] + window_size*(j+1))]
            windows[i * n_cells[1] + j] = img
            coords[i * n_cells[1] + j] = (margins[0] + window_size*i + window_size//2, margins[1] + window_size*j + window_size//2)

            # print((margins[0] + window_size*i),(margins[0] + window_size*(i+1)), (margins[1] + window_size*j),(margins[1] + window_size*(j+1)))
            # print(i * n_cells[1] + j, i, j, coords[i * n_cells[1] + j])
            # print(img.shape)
            # cv2.imshow('image1', img)
            # cv2.waitKey(0)



    return windows, coords


def patch_descriptors(image_windows):

    if image_windows.shape[1] != patch_size or image_windows.shape[2] != patch_size:
        resized_image_patches = np.empty((image_windows.shape[0], patch_size, patch_size, image_windows.shape[3]))
        for i in range(image_windows.shape[0]):
            resized_image_patches[i] = cv2.resize(image_windows[i], (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
    else:
        resized_image_patches = image_windows

    feats = feature_net.predict(resized_image_patches)

    reduced_feats = np.zeros((feats.shape[0], feats.shape[3]))

    for i in range(feats.shape[0]):

        patch_feats = feats[i] # 3x3x1280

        tot = np.zeros((patch_feats.shape[2],))

        for j in range(patch_feats.shape[0]):
            for k in range(patch_feats.shape[1]):
                tot = tot + patch_feats[j, k]

        avg = tot / (patch_feats.shape[0] * patch_feats.shape[1])

        reduced_feats[i] = avg

    return reduced_feats


def extract_features_to_db(image_path, image_name, scene_node):

    img_data = open_and_prepare_image(image_path)

    windows_a, coords_a = extract_windows(img_data, window_size, False, False)
    windows_b, coords_b = extract_windows(img_data, window_size, False, True)
    windows_c, coords_c = extract_windows(img_data, window_size, True, False)
    windows_d, coords_d = extract_windows(img_data, window_size, True, True)

    descriptors = patch_descriptors(np.concatenate((windows_a, windows_b, windows_c, windows_d), axis=0))
    coords = np.concatenate((coords_a, coords_b, coords_c, coords_d), axis=0)

    patches = []

    for i in range(descriptors.shape[0]):
        patches.append({'scene': image_name, 'size': window_size, 'loc': coords[i], 'des': descriptors[i]})

    insert_patches_result = database.insert_patches(patches)

    data = np.array([p['des'] for p in insert_patches_result], dtype=np.float32)
    data_labels = np.array([p['id'] for p in insert_patches_result])

    desc_index.add_items(data, data_labels)

    # insert neighbor relationships

    locations = np.array([p['loc'] for p in insert_patches_result], dtype=np.float32)

    loc_index = hnswlib.Index(space='l2', dim=2)
    loc_index.init_index(max_elements=len(insert_patches_result), ef_construction=200, M=50)
    loc_index.set_ef(50)
    loc_index.add_items(locations, data_labels)

    relationships = []

    for i in range(0, len(insert_patches_result)):
        from_label = data_labels[i]
        labels, distances = loc_index.knn_query(locations[i], k=5)
        for j in range(0, distances.shape[1]):
            if distances[0][j] > 0 and distances[0][j] <= window_size ** 2:
                relationships.append({"from": from_label, "to": labels[0][j], "dist": math.sqrt(distances[0][j])})

    result = database.insert_scene_neighbor_relationships(relationships)

    print("inserted neighbor relationships", len(result))

    # insert contains relationships

    scene_contains_relationships = [{"from": scene_node['id'], "to": label} for label in data_labels]
    result = database.insert_scene_contains_relationships(scene_contains_relationships)
    print("inserted contains relationships", len(result))

print('starting', len(images))

last_finished_index = -1
start = time.time()


for i in range(100):

    print('last_finished_index', last_finished_index)

    image_path = images[i][1]
    image_name = images[i][0]

    print(image_path, image_name)
    # insert scene node
    insert_scene_result = database.insert_scene({"scene": image_name})
    scene_node = insert_scene_result[0]
    print("inserted scene node")

    extract_features_to_db(image_path, image_name, scene_node)

    last_finished_index = i


end = time.time()
print('time elapsed', end - start)

print('saving desc_index')

desc_index.save_index("desc_index.bin")

print('finished')