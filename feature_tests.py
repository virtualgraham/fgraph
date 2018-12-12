import os
from os import listdir, path
import cv2
from feature_extractor import extract_features
import numpy as np

sun_rgbd_directory = "/Users/user/Desktop/SUNRGBD"

def list_image_paths(collection_dir):
    scene_dirs = [path.join(collection_dir, d, "image") for d in listdir(collection_dir) if path.isdir(path.join(collection_dir, d))]
    image_paths = [[path.join(d, e) for e in listdir(d)][0] for d in scene_dirs]
    return [(p[len(sun_rgbd_directory)+1:], p) for p in image_paths]

images = list_image_paths(path.join(sun_rgbd_directory, "kv2", "kinect2data")) + list_image_paths(path.join(sun_rgbd_directory, "kv2", "align_kv2"))

for i in range(3):
    test_image_path = images[i][1]
    test_image_name = images[i][0]
    print(test_image_path)
    patches = extract_features(test_image_path, test_image_name, 32)
    print(len(patches), patches[0])
