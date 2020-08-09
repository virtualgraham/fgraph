from keras.applications import mobilenetv2
from os import listdir, path
import numpy as np
import cv2
from scipy import spatial


patch_size = 96

images_directory = "/Users/user/Desktop/household_sample_patches"

feature_net = mobilenetv2.MobileNetV2(weights="imagenet", include_top=False, input_shape=(96, 96, 3))

images = [(e, path.join(images_directory, e)) for e in listdir(images_directory)]

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


windows = np.zeros((len(images), 192, 192, 3))

names = []
for i in range(len(images)):

    image_path = images[i][1]
    image_name = images[i][0]
    names.append(image_name)
    windows[i] = cv2.imread(image_path)

print('windows.shape', windows.shape)

des = patch_descriptors(windows)

print('des.shape', des.shape)

for i in range(len(des)):
    for j in range(len(des)):
        print(names[i], names[j], spatial.distance.cosine(des[i], des[j]))