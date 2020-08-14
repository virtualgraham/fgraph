import numpy as np
import cv2

from keras.applications import vgg16
from keras.applications.vgg16 import preprocess_input

from scipy import spatial

image_file = './media/flowers.jpg'


print("Starting...")

#initialize CNN and KNN index
model = vgg16.VGG16(weights="imagenet", include_top=False,  input_shape=(224, 224, 3))

image = cv2.imread(image_file)

print(image.shape)

# windows = np.empty((7*7, 224, 224, 3))

# for i in range(7):
#     for j in range(7):
#         windows[7*i + j] = image[(32*i):(32*i+224), (32*j):(32*j+224)]

window1 = image[100:324, 100:324]
window2 = image[132:(224+132), 132:(224+132)]


print(window1.shape)
print(window2.shape)

windows = np.array([window1, window2])

print(windows.shape)

windows_pp = preprocess_input(windows)
feats = model.predict(windows_pp)

print(feats.shape)

for i in range(1,6):
    for j in range(1,6):
        print(i, j, i-1,j-1, 1.0 - spatial.distance.cosine(feats[0,i,j], feats[1,i-1,j-1]))






