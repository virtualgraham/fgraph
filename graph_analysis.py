import os
import math
from os import listdir, path
from database import PatchGraphDatabase
import numpy as np
import cv2

sun_rgbd_directory = "/Users/user/Desktop/SUNRGBD"

def open_and_prepare_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def list_image_paths(collection_dir):
    scene_dirs = [path.join(collection_dir, d, "image") for d in listdir(collection_dir) if path.isdir(path.join(collection_dir, d))]
    image_paths = [[path.join(d, e) for e in listdir(d)][0] for d in scene_dirs]
    return [(p[len(sun_rgbd_directory)+1:], p) for p in image_paths]

images = list_image_paths(path.join(sun_rgbd_directory, "kv2", "kinect2data")) + list_image_paths(path.join(sun_rgbd_directory, "kv2", "align_kv2"))

database = PatchGraphDatabase()

def show_find_paths_between_scenes_result(result):

    image1_path = path.join(sun_rgbd_directory, result[0][0]['scene'])
    image2_path = path.join(sun_rgbd_directory, result[0][1]['scene'])

    image1 = open_and_prepare_image(image1_path)
    image2 = open_and_prepare_image(image2_path)

    for res in result:
        # print(res[0]['loc'], res[0]['size'], res[1]['loc'], res[1]['size'])

        s0 = res[0]['size']//2
        s1 = res[1]['size']//2

        loc0 = res[0]['loc']
        loc1 = res[1]['loc']

        cv2.rectangle(image1, (int(loc0[1] - s0), int(loc0[0] - s0)), (int(loc0[1] + s0), int(loc0[0] + s0)), (0, 255, 0), 1)
        cv2.rectangle(image2, (int(loc1[1] - s1), int(loc1[0] - s1)), (int(loc1[1] + s1), int(loc1[0] + s1)), (0, 255, 0), 1)

    cv2.imshow('image1',image1)
    cv2.waitKey(0)
    cv2.imshow('image2',image2)
    cv2.waitKey(0)


# for i in range(0,100):
#     print('.')
#     for j in range(0,100):
#         result = database.find_paths_between_scenes(images[i][0], images[j][0])
#         if len(result) > 50:
#             print(i, j)

# (1,82)(10,72)(20,9)(26,29)(26,75)(29,75)(40,72)(47,87)(55,70)


result = database.find_paths_between_scenes(images[29][0], images[75][0])
show_find_paths_between_scenes_result(result)