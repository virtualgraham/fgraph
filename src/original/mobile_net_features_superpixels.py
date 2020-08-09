from keras.applications import mobilenetv2
import numpy as np
import cv2
import hnswlib
from os import listdir, path
from database_v2 import PatchGraphDatabase
import time
import pymeanshift as pms

patch_size = 96
window_size = 96
channel_count = 3

patch_occupancy_threshold = .75

images_directory = "/Users/user/Desktop/household_images"

feature_net = mobilenetv2.MobileNetV2(weights="imagenet", include_top=False, input_shape=(96, 96, 3))

database = PatchGraphDatabase()

desc_index = hnswlib.Index(space='cosine', dim=1280)
desc_index.init_index(max_elements=7000000, ef_construction=500, M=32)
desc_index.set_ef(500)


images = [(e, path.join(images_directory, e)) for e in listdir(images_directory)]

def open_and_prepare_image(image_path):
    image = cv2.imread(image_path)
    # shrink image down 1088 x 816
    image = cv2.resize(image, (image.shape[1]//3, image.shape[0]//3), interpolation=cv2.INTER_CUBIC)
    return image


def extract_windows(image, superpixels, window_size=96, interior_w=False, interior_h=False):
    n_cells = (image.shape[0] // window_size, image.shape[1] // window_size)
    # print('n_cells', n_cells)

    if interior_w:
        n_cells = (n_cells[0] - 1, n_cells[1])

    if interior_h:
        n_cells = (n_cells[0], n_cells[1] - 1)

    img_shape = (n_cells[0] * window_size, n_cells[1] * window_size)

    margins = ((image.shape[0] - img_shape[0])//2, (image.shape[1] - img_shape[1])//2)

    superpixel_windows = np.zeros((n_cells[0] * n_cells[1], window_size, window_size))
    image_windows = np.zeros((n_cells[0] * n_cells[1], window_size, window_size, channel_count))
    coords = np.zeros((n_cells[0] * n_cells[1], 2))

    for i in range(n_cells[0]):
        for j in range(n_cells[1]):
            superpixel_window = superpixels[(margins[0] + window_size*i):(margins[0] + window_size*(i+1)), (margins[1] + window_size*j):(margins[1] + window_size*(j+1))]
            superpixel_windows[i * n_cells[1] + j] = superpixel_window

            image_window = image[(margins[0] + window_size*i):(margins[0] + window_size*(i+1)), (margins[1] + window_size*j):(margins[1] + window_size*(j+1))]
            image_windows[i * n_cells[1] + j] = image_window

            coords[i * n_cells[1] + j] = (margins[0] + window_size*i + window_size//2, margins[1] + window_size*j + window_size//2)

    return image_windows, superpixel_windows, coords


def get_patch_descriptors(image_windows):

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

def get_superpixel_connections(superpixels):
    connections = set()
    for i in range(superpixels.shape[0]):
        for j in range(superpixels.shape[1] - 1):
            if superpixels[i, j] == superpixels[i, j+1]:
                continue
            connections.add((superpixels[i, j], superpixels[i, j+1]))
            connections.add((superpixels[i, j+1], superpixels[i, j]))
    for j in range(superpixels.shape[1]):
        for i in range(superpixels.shape[0] - 1):
            if superpixels[i, j] == superpixels[i+1, j]:
                continue
            connections.add((superpixels[i, j], superpixels[i+1, j]))
            connections.add((superpixels[i+1, j], superpixels[i, j]))
    return connections

def extract_features_to_db(image_path, image_name, scene_node):

    img_data = open_and_prepare_image(image_path)

    (segmented_image, superpixels, number_regions) = pms.segment(img_data, spatial_radius=6, range_radius=4.5, min_density=50)

    # cv2.imshow('image', segmented_image)
    # cv2.waitKey(0)

    windows_a, sp_windows_a, coords_a = extract_windows(img_data, superpixels, window_size, False, False)
    windows_b, sp_windows_b, coords_b = extract_windows(img_data, superpixels, window_size, False, True)
    windows_c, sp_windows_c, coords_c = extract_windows(img_data, superpixels, window_size, True, False)
    windows_d, sp_windows_d, coords_d = extract_windows(img_data, superpixels, window_size, True, True)

    windows = np.concatenate((windows_a, windows_b, windows_c, windows_d), axis=0)
    sp_windows = np.concatenate((sp_windows_a, sp_windows_b, sp_windows_c, sp_windows_d), axis=0)
    coords = np.concatenate((coords_a, coords_b, coords_c, coords_d), axis=0)

    windows_f = []
    coords_f = []

    patch_contains_superpixels = []

    # filter out windows that are more than 2/3 occupied by a single superpixel
    for i in range(windows.shape[0]):

        counts = np.zeros((number_regions,))
        total = 0

        s = set()

        for sp in np.nditer(sp_windows[i]):
            total += 1
            counts[int(sp)] = counts[int(sp)] + 1
            s.add(int(sp))

        if np.amax(counts) > (total * patch_occupancy_threshold):
            continue

        patch_contains_superpixels.append(s)
        windows_f.append(windows[i])
        coords_f.append(coords[i])

    print("Discarding non-union windows:", len(windows) - len(windows_f), len(windows))

    windows = np.array(windows_f)
    coords = np.array(coords_f)

    # insert superpixels into database

    superpixel_dicts = []

    for i in range(number_regions):
        superpixel_dicts.append({'scene':image_name})

    insert_superpixels_result = database.insert_superpixels(superpixel_dicts)
    superpixel_db_ids = np.array([p['id'] for p in insert_superpixels_result])

    # insert patches into database

    descriptors = get_patch_descriptors(windows)

    patches = []

    for i in range(descriptors.shape[0]):
        patches.append({'scene': image_name, 'size': window_size, 'loc': coords[i], 'des': descriptors[i]})

    insert_patches_result = database.insert_patches(patches)

    patch_descriptors = np.array([p['des'] for p in insert_patches_result], dtype=np.float32)
    patch_db_ids = np.array([p['id'] for p in insert_patches_result])

    # add patch descriptors to index

    desc_index.add_items(patch_descriptors, patch_db_ids)

    # insert scene contains patch relationships

    scene_contains_patch_relationships = [{"from": scene_node['id'], "to": label} for label in patch_db_ids]
    result = database.insert_contains_relationships(scene_contains_patch_relationships)
    print("inserted scene contains patch relationships", len(result))

    # insert scene contains superpixel relationships

    scene_contains_superpixel_relationships = [{"from": scene_node['id'], "to": label} for label in superpixel_db_ids]
    result = database.insert_contains_relationships(scene_contains_superpixel_relationships)
    print("inserted scene contains superpixel relationships", len(result))

    # insert patch contains superpixel relationships

    patch_contains_superpixel_relationships = []
    for i in range(len(patch_contains_superpixels)):
        patch_superpixels = patch_contains_superpixels[i]
        patch_contains_superpixel_relationships.extend([{"from": patch_db_ids[i], "to": superpixel_db_ids[int(superpixel)]} for superpixel in patch_superpixels])
    result = database.insert_contains_relationships(patch_contains_superpixel_relationships)
    print("inserted patch contains superpixel relationships", len(result))

    # insert superpixel neighbor relationships

    superpixel_connections = get_superpixel_connections(superpixels)
    superpixel_neighbor_relationships = [{"from": superpixel_db_ids[int(a)], "to": superpixel_db_ids[int(b)]} for (a, b) in superpixel_connections]
    result = database.insert_neighbors_relationships(superpixel_neighbor_relationships)
    print("inserted superpixel neighbor relationships", len(result))

print('starting', len(images))

last_finished_index = -1
start = time.time()


for i in range(len(images)):

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