import cv2
import math
import numpy as np
from prev_approaches.l2_net import L2Net
from os import listdir, path
from prev_approaches.database import PatchGraphDatabase
import hnswlib
import time


def get_hog_gradients(image, cell_width=32):
    cell_size=(cell_width, cell_width)
    block_size = (2, 2) 
    nbins = 18  
    n_cells = (image.shape[0] // cell_size[0], image.shape[1] // cell_size[1])

    winSize = (image.shape[1] // cell_size[1] * cell_size[1], image.shape[0] // cell_size[0] * cell_size[0])
    blockSize = (block_size[1] * cell_size[1], block_size[0] * cell_size[0])
    blockStride = (cell_size[1], cell_size[0])
    cellSize = (cell_size[1], cell_size[0])

    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)

    hist = hog.compute(image)

    hog_feats = hist.reshape(n_cells[1] - block_size[1] + 1, n_cells[0] - block_size[0] + 1, block_size[0], block_size[1], nbins).transpose((1, 0, 2, 3, 4)) 

    gradients = np.zeros((n_cells[0], n_cells[1], nbins))

    # count cells (border cells appear less often across overlapping groups)
    cell_count = np.full((n_cells[0], n_cells[1], 1), 0, dtype=int)

    for off_y in range(block_size[0]):
        for off_x in range(block_size[1]):
            gradients[off_y:n_cells[0] - block_size[0] + off_y + 1, off_x:n_cells[1] - block_size[1] + off_x + 1] += hog_feats[:, :, off_y, off_x, :]
            cell_count[off_y:n_cells[0] - block_size[0] + off_y + 1, off_x:n_cells[1] - block_size[1] + off_x + 1] += 1

    # Average gradients
    gradients /= cell_count

    angles = np.zeros((n_cells[0], n_cells[1]))

    for x in range(gradients.shape[0]):
       for y in range(gradients.shape[1]): 
            
            h = gradients[x,y,:]
            n = np.argmax(gradients[x,y,:])

            # v = (n + 0.5) * (180/nbins)

            m = n - 1 if n > 0 else (nbins - 1)
            o = n + 1 if n < (nbins - 1) else 0

            p = h[m]
            q = h[n]
            r = h[o]
            
            if q > 0:
                s = p/(p+q) * (180/nbins)
                t = r/(q+r) * (180/nbins)
            else:
                print("not (p > 0 and q > 0)")
                s = 0
                t = 0

            u = (n + 0.5) * (180/nbins)
            v = u - s + t
            # print(m,n,o,p,q,r,s,t,u,v)

            if v >= 90:
                angles[x,y] = 180 - v
            else:
                angles[x,y] = 0 - v

    return angles


def extract_patches(image, cellSize=32, interior_w=False, interior_h=False):

    n_cells = (image.shape[0] // cellSize, image.shape[1] // cellSize)
    rotation_size = math.ceil(math.sqrt(2*cellSize**2)) + 2
    rotation_margin = math.ceil((rotation_size-cellSize) / 2) * 2

    if (image.shape[0] - n_cells[0]*cellSize) < rotation_margin:
        n_cells = (n_cells[0] - 1, n_cells[1])
        
    if (image.shape[1] - n_cells[1]*cellSize) < rotation_margin:
        n_cells = (n_cells[0], n_cells[1]-1)

    if interior_w:
        n_cells = (n_cells[0]-1, n_cells[1])

    if interior_h:
        n_cells = (n_cells[0], n_cells[1]-1)

    img_shape = (n_cells[0] * cellSize, n_cells[1] * cellSize)
    margins = (image.shape[0] - img_shape[0], image.shape[1] - img_shape[1])

    img = image[(margins[0]//2):(image.shape[0]-margins[0]+margins[0]//2), (margins[1]//2):(image.shape[1]-margins[1]+margins[1]//2)]

    # print('img.shape', img.shape, (n_cells[0]*cellSize, n_cells[1]*cellSize))

    angles = get_hog_gradients(img, cellSize)

    patches = np.zeros((n_cells[0], n_cells[1], cellSize, cellSize), np.uint8)

    coords = np.zeros((n_cells[0], n_cells[1], 2))

    for x in range(angles.shape[0]):
        for y in range(angles.shape[1]): 
            # the point in the original image that represents the center of the cell
            center = (x * cellSize + cellSize//2 + margins[0]//2, y * cellSize + cellSize//2 + margins[1]//2)
            coords[x,y] = center
            top_left = (center[0] - rotation_size//2, center[1] - rotation_size//2)
            rotation_patch = image[top_left[0]:top_left[0]+rotation_size, top_left[1]:top_left[1]+rotation_size]
            M = cv2.getRotationMatrix2D((rotation_size/2,rotation_size/2),-angles[x,y],1)
            dst = cv2.warpAffine(rotation_patch,M,(rotation_size,rotation_size))
            rotated_patch = dst[((rotation_size-cellSize)//2):((rotation_size-cellSize)//2+cellSize), ((rotation_size-cellSize)//2):((rotation_size-cellSize)//2+cellSize)]
            patches[x, y] = rotated_patch

    return np.reshape(patches,(n_cells[0] * n_cells[1], cellSize, cellSize)), np.reshape(angles, (n_cells[0] * n_cells[1],)), np.reshape(coords, (n_cells[0] * n_cells[1], 2)), n_cells

descriptor_size = 256
descriptor_window_size = 64
l2_net = L2Net("L2Net-HP+", True)


def extract_image_features(image_path, image_name, window_size=32):
    image = cv2.imread(image_path,0)

    # print('image.dtype', image.dtype)

    patches_a, angles_a, coords_a, n_cells_a = extract_patches(image, window_size, False, False)
    patches_b, angles_b, coords_b, n_cells_b = extract_patches(image, window_size, False, True)
    patches_c, angles_c, coords_c, n_cells_c = extract_patches(image, window_size, True, False)
    patches_d, angles_d, coords_d, n_cells_d = extract_patches(image, window_size, True, True)

    ###########
    # TESTING #
    ###########

    # n_cells = n_cells_a
    # patches = patches_a

    # assembly = np.zeros((n_cells[0] * window_size, n_cells[1] * window_size), dtype=np.uint8)
    
    # for i in range(n_cells[0]):
    #     for j in range(n_cells[1]):
    #         patch = np.reshape(patches[i*n_cells[1] + j], (window_size,window_size))
    #         assembly[i*window_size:(i+1)*window_size, j*window_size:(j+1)*window_size] = patch

    # cv2.imshow("assembly", assembly)
    # cv2.waitKey(0)

    ###########
    ###########

    patches = np.concatenate((patches_a, patches_b, patches_c, patches_d), axis=0)
    coords = np.concatenate((coords_a, coords_b, coords_c, coords_d), axis=0)
    angles = np.concatenate((angles_a, angles_b, angles_c, angles_d), axis=0)

    if window_size == descriptor_window_size:
        patches_resized = np.reshape(patches, (patches.shape[0], descriptor_window_size, descriptor_window_size, 1))
    else:
        patches_resized = np.empty((patches.shape[0], descriptor_window_size, descriptor_window_size, 1))
        for i in range(patches.shape[0]):
            patch_resized = cv2.resize(patches[i], (descriptor_window_size, descriptor_window_size), interpolation=cv2.INTER_CUBIC)
            patches_resized[i] = np.reshape(patch_resized, (descriptor_window_size, descriptor_window_size, 1))

    descriptors = l2_net.calc_descriptors(patches_resized)

    # print('descriptors.shape', descriptors.shape)
    # print('coords.shape', coords.shape)
    # print('angles.shape', angles.shape)

    patch_dicts = []

    for i in range(patches.shape[0]):
        patch_dict = {'scene': image_name, 'size': window_size, 'loc': (coords[i][0], coords[i][1]), 'des': descriptors[i], 'angle':angles[i]}
        patch_dicts.append(patch_dict)

    return patch_dicts


sun_rgbd_directory = "/Users/user/Desktop/SUNRGBD"


def list_image_paths(collection_dir):
    scene_dirs = [path.join(collection_dir, d, "image") for d in listdir(collection_dir) if path.isdir(path.join(collection_dir, d))]
    image_paths = [[path.join(d, e) for e in listdir(d)][0] for d in scene_dirs]
    return [(p[len(sun_rgbd_directory)+1:], p) for p in image_paths]


images = list_image_paths(path.join(sun_rgbd_directory, "kv2", "kinect2data")) + list_image_paths(path.join(sun_rgbd_directory, "kv2", "align_kv2"))

database = PatchGraphDatabase()

desc_index = hnswlib.Index(space = 'l2', dim = descriptor_size)
desc_index.init_index(max_elements = 7000000, ef_construction = 200, M = 16)
desc_index.set_ef(50)


def extract_features_to_db(image_path, image_name, scene_node, size):
    patches = extract_image_features(image_path, image_name, size)

    insert_patches_result = database.insert_patches(patches)
    print("inserted patch nodes", len(insert_patches_result))

    data = np.array([p['des'] for p in insert_patches_result], dtype=np.float32)
    data_labels = np.array([p['id'] for p in insert_patches_result])

    desc_index.add_items(data, data_labels)

    # insert neighbor relationships

    locations = np.array([p['loc'] for p in insert_patches_result], dtype=np.float32)

    loc_index = hnswlib.Index(space = 'l2', dim = 2)
    loc_index.init_index(max_elements = len(insert_patches_result), ef_construction = 1000, M = 50)
    loc_index.set_ef(1000)
    loc_index.add_items(locations, data_labels)

    relationships = []

    for i in range(0, len(insert_patches_result)):
        from_label = data_labels[i]
        labels, distances = loc_index.knn_query(locations[i], k=13)
        for j in range(0, distances.shape[1]):
            if distances[0][j] > 0 and distances[0][j] <= size**2:
                relationships.append({"from": from_label, "to": labels[0][j], "dist": math.sqrt(distances[0][j])})

    result = database.insert_scene_neighbor_relationships(relationships)

    print("inserted neighbor relationships", len(result))

    # insert contains relationships

    scene_contains_relationships = [{"from": scene_node['id'], "to": label} for label in data_labels]
    result = database.insert_scene_contains_relationships(scene_contains_relationships) 
    print("inserted contains relationships", len(result))


last_finished_index = -1
start = time.time()

for i in range(100):

    print('last_finished_index', last_finished_index)

    image_path = images[i][1]
    image_name = images[i][0]

    # insert scene node
    insert_scene_result = database.insert_scene({"scene": image_name})
    scene_node = insert_scene_result[0]
    print("inserted scene node")

    extract_features_to_db(image_path, image_name, scene_node, 64)
    extract_features_to_db(image_path, image_name, scene_node, 32)

    last_finished_index = i
    
end = time.time()
print('time elapsed', end - start)

print('saving desc_index')

desc_index.save_index("desc_index.bin")

print('finished')