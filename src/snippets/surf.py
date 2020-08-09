
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time


img = cv2.imread('./media/vlcsnap-2020-08-07-14h09m55s584.png',0)

# Initiate STAR detector
orb = cv2.ORB_create( nfeatures=10, fastThreshold=7)


# find the keypoints with ORB
kp = orb.detect(img,None)

print(kp)

