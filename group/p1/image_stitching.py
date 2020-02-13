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
    def __init__(self, im_l, im_r, animation=True, alg='sift', nfeatures=1500, ransac_iter=1000):
        self.a = animation
        self.ratio = 0.6
        self.waittime = 50
        self.im_l = cv2.imread(im_l)
        self.im_r = cv2.imread(im_r)
        self.gray_l, self.gray_r = cv2.cvtColor(self.im_l, cv2.COLOR_BGR2GRAY), cv2.cvtColor(self.im_r, cv2.COLOR_BGR2GRAY)
        if alg == 'sift':
            self.alg = cv2.xfeatures2d.SIFT_create()
        if alg == 'surf':
            self.alg = cv2.xfeatures2d.SURF_create()
        if alg == 'orb':
            self.alg = cv2.ORB_create(nfeatures=nfeatures)
        self.keypoints_l, self.descriptors_l = self.alg.detectAndCompute(self.gray_l, None)
        self.keypoints_r, self.descriptors_r = self.alg.detectAndCompute(self.gray_r, None)

        # TODO try cv2.DIST_L2 instead
        matcher = cv2.BFMatcher(cv2.NORM_L2)

        matches1 = matcher.knnMatch(self.descriptors_l, self.descriptors_r, k=2)
        matches2 = matcher.knnMatch(self.descriptors_r, self.descriptors_l, k=2)

        _, matches1 = self.ratio_test(matches1)
        _, matches2 = self.ratio_test(matches2)

        sym_matches = self.symmetry_test(matches1, matches2)

        shape_l, shape_r = self.im_l.shape, self.im_r.shape
        self.extended = [max(shape_l[i], shape_r[i]) for i in range(len(shape_l))]
        self.denoised = self.ransac(sym_matches, ransac_iter)
        pickle.dump(self.denoised, open("points", "wb"))


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

    def distance(self, p1, p2):
        return ((p1.pt[0]-p2.pt[0])**2 + (p1.pt[1]-p2.pt[1])**2) ** 0.5

    def position(self, kps):
        return cv2.KeyPoint.convert([(kp.pt[0]+self.extended[1], kp.pt[1]) for kp in kps])

    def color(self):
        return random.randint(0, 255), random.randint(0, 255), random.randint(0,255)

    def evaluate(self, slope_rate, distance_rate, p1, p2, error):
        lower, upper = 1-error, 1+error
        color_l, color_r = self.gray_l[int(p1.pt[1])][int(p1.pt[0])], self.gray_r[int(p2.pt[1])][int(p2.pt[0])]
        if lower < slope_rate < upper and lower < distance_rate < upper and \
            abs(p1.angle+p2.angle)/360 < error and \
            abs(color_l-color_r)/255 < error:
            return True
        return False

    def intTuple(self, t):
        return tuple([int(i) for i in t])

    def inlinerAnimation(self, im, iter, p1, p2, n):
        cp1, cp2 = [p1], self.position([p2])
        im = cv2.drawKeypoints(im, cp1 + cp2, None)
        for i in range(len(cp1)):
            im = cv2.line(im, self.intTuple(cp1[i].pt), self.intTuple(cp2[i].pt), self.color(), 1)
            cv2.imshow('%d iteration best inliners:%d' % (iter, n), im)
            cv2.waitKey(self.waittime)
        return im

    def ransac(self, sym, iter, error=0.2, thresh_size=0.3):
        kl = [self.keypoints_l[i.queryIdx] for i in sym]
        kr = [self.keypoints_r[i.trainIdx] for i in sym]
        print("Symmetric points len:%d %d" % (len(kl), len(kr)))
        best_match = 0
        best_inliners = [[],[]]
        for i in range(iter):
            lp, rp = self.im_l, self.im_r
            p1, p2 = random.choice(kl), random.choice(kr)
            kp_l, kp_r = [p1], [p2]
            if self.a:
                kp_r = self.position(kp_r)
                showimg = cv2.hconcat([np.resize(lp, self.extended), np.resize(rp, self.extended)])
                showimg = cv2.drawKeypoints(showimg, kp_l+kp_r, None)
                cv2.destroyAllWindows()
                for k in range(1):
                    ptl, ptr = tuple([int(j) for j in kp_l[k].pt]), tuple([int(j) for j in kp_r[k].pt])
                    showimg = cv2.line(showimg, ptl, ptr, self.color(), 3)
                    cv2.imshow('%d iteration best inliners:%d' % (i, best_match), showimg)
                    cv2.waitKey(self.waittime)
            slope = (p2.pt[1] - p1.pt[1]) / (p2.pt[0] - p1.pt[0])
            distance = self.distance(p1, p2)
            kpl_set, kpr_set = {p1.pt}, {p2.pt}
            cur_inliners = [[p1],[p2]]
            for j in kl:
                if j.pt in kpl_set: continue
                for k in kr:
                    if k.pt in kpr_set: continue
                    cur_slope = (j.pt[1] - k.pt[1]) / (j.pt[0] - k.pt[1])
                    cur_distance = self.distance(j, k)
                    if self.evaluate(cur_slope/slope, cur_distance/distance, j, k, error):
                        kpl_set.add(j.pt)
                        kpr_set.add(k.pt)
                        if self.a: showimg = self.inlinerAnimation(showimg, i, j, k, best_match)
                        cur_inliners[0].append(j)
                        cur_inliners[1].append(k)
                        break
            if len(kpl_set) > best_match:
                best_inliners = cur_inliners
                best_match = len(kpl_set)
                print('Find a better match:%d', best_match)
            if best_match > (len(kl) *(1-error)) * thresh_size:
                break
        return best_inliners

s = STITCHING('mt_l.png', 'mt_r.png')
