
import numpy as np
import cv2
from matplotlib import pyplot as plt

window_size = 32 + 64

def extract_window(frame, pos):
    half_w = window_size/2.0
    bottom_left = [int(round(pos[0]-half_w)), int(round(pos[1]-half_w))]
    top_right = [bottom_left[0]+window_size, bottom_left[1]+window_size]
    if bottom_left[0] < 0:
        #print("bottom_left[0] < 0")
        top_right[0] -= bottom_left[0]
        bottom_left[0] = 0
    if bottom_left[1] < 0:
        #print("bottom_left[1] < 0")
        top_right[1] -= bottom_left[1]
        bottom_left[1] = 0
    if top_right[0] >= frame.shape[0]:
        #print("top_right[0] >= frame.shape[0]")
        bottom_left[0] -= (top_right[0]-frame.shape[0]+1)
        top_right[0] = frame.shape[0]-1
    if top_right[1] >= frame.shape[1]:
        #print("top_right[1] >= frame.shape[1]")
        bottom_left[1] -= (top_right[1]-frame.shape[1]+1)
        top_right[1] = frame.shape[1]-1
        

    # print(pos, bottom_left, top_right)
    return frame[bottom_left[0]:top_right[0], bottom_left[1]:top_right[1]]

img = cv2.imread('./media/vlcsnap-2020-08-07-12h49m55s368.png',0)

win = extract_window(img, (277,356))



# Initiate STAR detector
orb = cv2.ORB_create( nfeatures=10)


# find the keypoints with ORB
kp = orb.detect(win,None)

# compute the descriptors with ORB
kp, des = orb.compute(win, kp)

# # draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(win, kp, None, color=(0,255,0), flags=0)
plt.imshow(img2), plt.show()