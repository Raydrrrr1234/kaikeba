##
# Data: 2020/2/10 Mon
# Author: Ruikang Dai
# Description: Kaikeba homework week 4 group project
##

import random
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import cv2

class STITCHING(object):
    def __init__(self, im_l, im_r, animation=True, alg='orb', nfeatures=1500, ransac_iter=100):
        self.a = animation
        self.im_l = cv2.imread(im_l)
        self.im_r = cv2.imread(im_r)
        gray_l, gray_r = cv2.cvtColor(self.im_l, cv2.COLOR_BGR2GRAY), cv2.cvtColor(self.im_r, cv2.COLOR_BGR2GRAY)
        if alg == 'sift':
            self.alg = cv2.xfeatures2d.SIFT_create()
        if alg == 'surf':
            self.alg = cv2.xfeatures2d.SURF_create()
        if alg == 'orb':
            self.alg = cv2.ORB_create(nfeatures=nfeatures)
        keypoints_l, descriptors_l = self.alg.detectAndCompute(gray_l, None)
        keypoints_r, descriptors_r = self.alg.detectAndCompute(gray_r, None)
        shape_l, shape_r = self.im_l.shape, self.im_r.shape
        extended = [max(shape_l[i], shape_r[i]) for i in range(len(shape_l))]
        self.extended_l = [extended[i]-shape_l[i] for i in range(len(shape_l))]
        self.extended_r = [extended[i]-shape_r[i] for i in range(len(shape_r))]
        self.showimg = cv2.hconcat(np.resize(self.im_l), np.resize(self.im_r))
        self.ransac(keypoints_l, descriptors_l, keypoints_r, descriptors_r, ransac_iter)

    def ransac(self, kl, dl, kr, dr, iter):
        for i in range(iter):
            if self.a:
                cv2.imshow()
