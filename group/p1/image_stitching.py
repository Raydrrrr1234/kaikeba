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
import pickle

class STITCHING(object):

    def __init__(self, im_l, im_r, alg='sift', nfeatures=1500, ransac_iter=100, thread_num=2):
        self.ratio = 0.6
        self.im_l = cv2.imread(im_l)
        self.im_r = cv2.imread(im_r)
        self.gray_l, self.gray_r = cv2.cvtColor(self.im_l, cv2.COLOR_BGR2GRAY), cv2.cvtColor(self.im_r,
                                                                                             cv2.COLOR_BGR2GRAY)
        if alg == 'sift':
            self.alg = cv2.xfeatures2d.SIFT_create()
        if alg == 'surf':
            self.alg = cv2.xfeatures2d.SURF_create()
        if alg == 'orb':
            self.alg = cv2.ORB_create(nfeatures=nfeatures)
        self.keypoints_l, self.descriptors_l = self.alg.detectAndCompute(self.gray_l, None)
        self.keypoints_r, self.descriptors_r = self.alg.detectAndCompute(self.gray_r, None)

        matcher = cv2.BFMatcher(cv2.NORM_L2)

        matches1 = matcher.knnMatch(self.descriptors_l, self.descriptors_r, k=2)
        matches2 = matcher.knnMatch(self.descriptors_r, self.descriptors_l, k=2)

#        _, matches1 = self.ratio_test(matches1)
#        _, matches2 = self.ratio_test(matches2)

        sym_matches = self.symmetry_test(matches1, matches2)
        if len(sym_matches) < 30:
            return
        kl = [self.keypoints_l[i.queryIdx] for i in sym_matches]
        kr = [self.keypoints_r[i.trainIdx] for i in sym_matches]
        shape_l, shape_r = self.im_l.shape, self.im_r.shape
        ptsr, ptsl = np.float32([i.pt for i in kr]), np.float32([i.pt for i in kl])
        M, mask = cv2.findHomography(ptsr, ptsl, cv2.RANSAC, 5.0)
        h, w, chnn = self.im_r.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        self.dst = cv2.warpPerspective(self.im_r, M, (self.im_l.shape[1] + self.im_r.shape[1], self.im_l.shape[0]))
        self.dst[0:self.im_l.shape[0], 0:self.im_l.shape[1]] = self.im_l
        cv2.imshow('Img', self.dst)
        cv2.waitKey(0)

    def getStitching(self):
        return self.dst

    def symmetry_test(self, matches1, matches2):
        symmetric_matches = []
        for match1 in matches1:
            if len(match1) < 2:
                continue
            for match2 in matches2:
                if len(match2) < 2:
                    continue
                if match1[0].queryIdx == match2[0].trainIdx and match2[0].queryIdx == match1[0].trainIdx:
                    symmetric_matches.append(cv2.DMatch(match1[0].queryIdx, match1[0].trainIdx, match1[0].distance))
                    break

        return symmetric_matches

    def ratio_test(self, matches):
        removed = 0
        for match in matches:
            if len(match) > 1:
                if match[0].distance / match[1].distance > self.ratio:
                    match.clear()
                    removed += 1
            else:
                match.clear()
                removed += 1
        return removed, matches


s = STITCHING('lenna50l.jpg', 'lenna50r.jpg')
#print(s.getStitching())
