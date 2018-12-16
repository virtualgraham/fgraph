import os
import math
from os import listdir, path
from database import PatchGraphDatabase
import numpy as np
import cv2
from random import randint

image_size = 224
# sun_rgbd_directory = "/Users/user/Desktop/SUNRGBD"
images_directory = "/Users/user/Desktop/household_images"

def get_scaled_dims(orig_dims, new_smallest_dim):
    if orig_dims[0] < orig_dims[1]:
        new_larger_dim = (new_smallest_dim/orig_dims[0]) * orig_dims[1]
        return int(new_larger_dim), int(new_smallest_dim)
    else:
        new_larger_dim = (new_smallest_dim/orig_dims[1]) * orig_dims[0]
        return int(new_smallest_dim), int(new_larger_dim)


def crop_to_square(image):
    diff = abs(image.shape[1] - image.shape[0])
    diff = diff//2
    if image.shape[0] < image.shape[1]:
        #print('image.shape[0] < image.shape[1]', image.shape[0], image.shape[1])
        return image[:, diff:(diff+image.shape[0])]
    elif image.shape[1] < image.shape[0]:
        #print('image.shape[1] < image.shape[0]', image.shape[1], image.shape[0])
        return image[diff:(diff+image.shape[1]), :]
    else:
        #print('else')
        return image


def open_and_prepare_image(image_path):
    image = cv2.imread(image_path)
    return image


images = [(e, path.join(images_directory, e)) for e in listdir(images_directory)]


database = PatchGraphDatabase()

def show_find_paths_between_scenes_result(result):

    image1_path = path.join(images_directory, result[0][0]['scene'])
    image2_path = path.join(images_directory, result[0][1]['scene'])

    image1 = open_and_prepare_image(image1_path)
    image2 = open_and_prepare_image(image2_path)

    for res in result:
        # print(res[0]['loc'], res[0]['size'], res[1]['loc'], res[1]['size'])

        s0 = res[0]['size']//2
        s1 = res[1]['size']//2

        loc0 = res[0]['loc']
        loc1 = res[1]['loc']

        color = (randint(0, 255), randint(0, 255), randint(0, 255))

        cv2.rectangle(image1, (int(loc0[1] - s0), int(loc0[0] - s0)), (int(loc0[1] + s0), int(loc0[0] + s0)), color, 5)
        cv2.rectangle(image2, (int(loc1[1] - s1), int(loc1[0] - s1)), (int(loc1[1] + s1), int(loc1[0] + s1)), color, 5)



    image1 = cv2.resize(image1, (640, 480))
    image2 = cv2.resize(image2, (640, 480))

    cv2.imshow('image1', image1)
    cv2.waitKey(0)
    cv2.imshow('image2', image2)
    cv2.waitKey(0)


# for i in range(0,100):
#     print('.')
#     for j in range(0,100):
#         result = database.find_paths_between_scenes(images[i][0], images[j][0])
#         if len(result) > 50:
#             print(i, j)

# (1,82)(10,72)(20,9)(26,29)(26,75)(29,75)(40,72)(47,87)(55,70)


for i in range(21,100):
    for j in range(20,100):
        result = database.find_paths_between_scenes(images[i][0], images[j][0])
        if len(result) > 0:
            break
    if len(result) > 0:
        break

print(i,j)

show_find_paths_between_scenes_result(result)