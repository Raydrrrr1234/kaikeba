import random
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import cv2

img = cv2.imread("lenna50.jpg", cv2.IMREAD_GRAYSCALE)
sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()
orb = cv2.ORB_create(nfeatures=1500)

keypoints_sift, descriptors_sift = sift.detectAndCompute(img, None)
keypoints_surf, descriptors_surf = surf.detectAndCompute(img, None)
keypoints_orb, descriptors_orb = orb.detectAndCompute(img, None)

# Brute Force Matching
'''
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)
matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
'''

img = cv2.drawKeypoints(img, keypoints_sift, None)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.drawKeypoints(img, keypoints_surf, None)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.drawKeypoints(img, keypoints_orb, None)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()