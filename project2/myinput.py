import argparse
import math
import cv2
import numpy as np
from PIL import Image
from mydata import RandomFlipLR, RandomRotate


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def main_test():
    parser = argparse.ArgumentParser(description='Detector')
    parser.add_argument('--test', type=str, default='', required=True,
                        help='test image')
    parser.add_argument('--landmarks', type=str, default='',
                        help='image landmarks example: 2,3,4,5')
    args = parser.parse_args()

    image = Image.open(args.test).convert("RGB")
    landmarks = np.float32([float(i) for i in args.landmarks.split()])
    sample = {'image': image, 'landmarks': landmarks, 'net': '', 'angle': 30}
    img = np.array(sample['image'])
    kps = [cv2.KeyPoint(i, j, 1) for (i, j) in zip(sample['landmarks'][::2], sample['landmarks'][1::2])]
    cv2.drawKeypoints(img, kps, img, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('Original', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    rlr = RandomFlipLR()
    sample = rlr(sample)
    img = np.array(sample['image'])
    kps = [cv2.KeyPoint(i, j, 1) for (i, j) in zip(sample['landmarks'][::2], sample['landmarks'][1::2])]
    cv2.drawKeypoints(img, kps, img, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('RandomFlipLR', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(img.shape)
    cv2.waitKey(0)
    rr = RandomRotate()
    sample = rr(sample)
    img = np.array(sample['image'])
    kps = [cv2.KeyPoint(i, j, 1) for (i, j) in zip(sample['landmarks'][::2], sample['landmarks'][1::2])]
    cv2.drawKeypoints(img, kps, img, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('RandomRotate', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(img.shape)
    cv2.waitKey(0)


if __name__ == '__main__':
    main_test()