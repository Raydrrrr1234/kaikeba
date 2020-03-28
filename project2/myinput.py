import argparse
import math
import cv2
import numpy as np
import torch
from PIL import Image
from mydata import RandomFlip, RandomRotate, RandomNoise
from torchvision import transforms
from myDataS3 import get_train_test_set_w_err, RandomErasing


def test_sample(sample):
    img = np.array(sample['image'])
    kps = [cv2.KeyPoint(i, j, 1) for (i, j) in zip(sample['landmarks'][::2], sample['landmarks'][1::2])]
    for i in kps:
        cv2.drawKeypoints(img, [i], img, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('Original', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
    rlr = RandomFlip()
    sample = rlr(sample)
    img = np.array(sample['image'])
    kps = [cv2.KeyPoint(i, j, 1) for (i, j) in zip(sample['landmarks'][::2], sample['landmarks'][1::2])]
    cv2.drawKeypoints(img, kps, img, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('RandomFlipLR', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    rr = RandomRotate()
    sample = rr(sample)
    img = np.array(sample['image'])
    kps = [cv2.KeyPoint(i, j, 1) for (i, j) in zip(sample['landmarks'][::2], sample['landmarks'][1::2])]
    cv2.drawKeypoints(img, kps, img, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('RandomRotate', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    re = RandomErasing(p=1)
    img = np.array(sample['image'])
    img = img.transpose((2, 0, 1))
    sample['image'] = torch.from_numpy(img)
    sample = re(sample)
    img = np.array(sample['image']).transpose((1, 2, 0))
    cv2.imshow('RandomRotate', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    rn = RandomNoise()
    img = np.array(sample['image'])
    img = img.transpose((2, 0, 1))
    sample['image'] = torch.from_numpy(img)
    sample = rn(sample)
    img = np.array(sample['image']).transpose((1, 2, 0))
    cv2.imshow('RandomRotate', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


def main_test(args):
    train_set, test_set = get_train_test_set_w_err(args['net'], args['roi'], args['angle'])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size)

    train_iter = iter(train_loader)
    valid_iter = iter(valid_loader)

    sample = next(train_iter)
    test_sample(sample)
    sample = next(valid_iter)
    test_sample(sample)


if __name__ == '__main__':
    args = {'net': '', 'roi': 0.25, 'angle': 15}
    main_test(args)
