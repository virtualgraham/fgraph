import cv2
import pymeanshift as pms

original_image = cv2.imread("example.png")

(segmented_image, labels_image, number_regions) = pms.segment(original_image, spatial_radius=6, 
                                                              range_radius=4.5, min_density=50)