import random
import warnings

from torch.utils.data import Dataset
import torch
from PIL import Image
import cv2
import math
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as F

path = 'data'
folder_list = ['I', 'II']

# train_boarder = 224

need_record = False

train_list = 'train.txt'
test_list = 'test.txt'


def channel_norm(img):
    # img: ndarray, float32
    mean = np.mean(img)
    std = np.std(img)
    pixels = (img - mean) / (std + 0.0000001)
    return pixels


def warp(orig_shape, landmarks, width=112, height=112):
    pts = [[max(0, i), max(0, j)] for (i, j) in zip(landmarks[::2], landmarks[1::2])]
    orig_h, orig_w = orig_shape[:2]
    o = np.float32([[0, 0], [orig_w, 0], [0, orig_h], [orig_w, orig_h]])
    w = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    M = cv2.getPerspectiveTransform(o, w)
    w_pts = cv2.perspectiveTransform(np.float32([pts]), M)
    return w_pts.flatten()


# 按行读取数据为 img_name, rect, landmarks
def my_parse_line(line):
    line_parts = line.strip().split()
    img_name = line_parts[0]
    rect = list(map(int, list(map(float, line_parts[1:5]))))
    landmarks = list(map(float, line_parts[5: len(line_parts)]))
    return img_name, rect, landmarks


# 按照ratio扩脸选框
def my_expand_roi(rect, img_width, img_height, ratio=0.25):
    if rect[0] > rect[2] and rect[1] > rect[3]:
        return my_expand_roi([rect[2:], rect[:2]], img_height, img_height, ratio=ratio)
    width = int((rect[2] - rect[0]) * ratio)
    height = int((rect[3] - rect[1]) * ratio)
    return max(0, rect[0] - width), max(0, rect[1] - height), min(rect[2] + width, img_width - 1), min(rect[3] + height,
                                                                                                       img_height - 1)


class Normalize(object):
    """
        Resize to train_boarder x train_boarder.
        Then do channel normalization: (image - mean) / std_variation
    """

    def __call__(self, sample):
        image, landmarks, net = sample['image'], sample['landmarks'], sample['net']
        if net == '':
            train_boarder = 112
        else:
            train_boarder = 224
        # Resize image
        image_resize = np.asarray(image.resize((train_boarder, train_boarder), Image.BILINEAR), dtype=np.float32)
        # landmarks 对应缩放变换
        landmarks = warp((image.height, image.width), landmarks, width=train_boarder, height=train_boarder)
        # Normalization
        image = channel_norm(image_resize)
        return {'image': image,
                'landmarks': landmarks,
                'net': net}


class RandomRotate(object):
    """
        Rotate the picture small angle
    """

    def __call__(self, sample):
        image, landmarks, net, angle = sample['image'], sample['landmarks'], sample['net'], sample['angle']
        a0 = random.random() * min(angle, 30) - 15
        a1, a2 = angle, angle * math.pi / 180
        ox, oy = image.width // 2, image.height // 2
        image = image.rotate(-a1, Image.BILINEAR, expand=0)
        cur = [[ox + math.cos(a2) * (i - ox) - math.sin(a2) * (j - oy),
                oy + math.sin(a2) * (i - ox) + math.cos(a2) * (j - oy)]
               for (i, j) in zip(sample['landmarks'][::2], sample['landmarks'][1::2])]
        landmarks = np.float32(cur).flatten()
        return {'image': image,
                'landmarks': landmarks,
                'net': net,
                'angle': angle}


class RandomFlip(object):
    """
        Randomly flip left and right
    """
    def __call__(self, sample):
        image, landmarks, net, angle = sample['image'], sample['landmarks'], sample['net'], sample['angle']
        # Flip image randomly
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            landmarks = np.array(
                [landmarks[i] if i % 2 == 1 else image.width - landmarks[i] for i in range(len(landmarks))])
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            landmarks = np.array([landmarks[i] if i % 2 == 0 else image.height-landmarks[i] for i in range(len(landmarks))])
        return {'image': image,
                'landmarks': landmarks,
                'net': net,
                'angle': angle}


class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        Tensors channel sequence: N x C x H x W
    """

    def __call__(self, sample):
        image, landmarks, net = sample['image'], sample['landmarks'], sample['net']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if net == '':
            image = np.expand_dims(image, axis=0)
        else:
            image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks),
                'net': net}


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         p: probability that the random erasing operation will be performed.
         scale: range of proportion of erased area against input image.
         ratio: range of aspect ratio of erased area.
         value: erasing value. Default is 0. If a single int, it is used to
            erase all pixels. If a tuple of length 3, it is used to erase
            R, G, B channels respectively.
            If a str of 'random', erasing each pixel with random values.
         inplace: boolean to make this transform inplace. Default set to False.

    Returns:
        Erased Image.
    # Examples:
        >>> transform = transforms.Compose([
        >>> transforms.RandomHorizontalFlip(),
        >>> transforms.ToTensor(),
        >>> transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> transforms.RandomErasing(),
        >>> ])
    """

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):

        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("range of scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("range of random erasing probability should be between 0 and 1")

        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace

    @staticmethod
    def get_params(img, scale, ratio, value=0):
        """Get parameters for ``erase`` for a random erasing.

        Args:
            img (Tensor): Tensor image of size (C, H, W) to be erased.
            scale: range of proportion of erased area against input image.
            ratio: range of aspect ratio of erased area.
            value: default value to fill
        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        img_c, img_h, img_w = img.shape
        area = img_h * img_w

        for attempt in range(10):
            erase_area = random.uniform(scale[0], scale[1]) * area
            aspect_ratio = random.uniform(ratio[0], ratio[1])

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))

            if h < img_h and w < img_w:
                i = random.randint(0, img_h - h)
                j = random.randint(0, img_w - w)
                v = value
                return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, img

    def __call__(self, sample):
        """
        Args:
            sample dict type
            sample['image']: (Tensor): Tensor image of size (C, H, W) to be erased.
            sample['landmarks']: flatten points information

        Returns:
            sample (Tensor): Erased Tensor image.
        """
        image, landmarks, net = sample['image'], sample['landmarks'], sample['net']

        if random.uniform(0, 1) < self.p:
            x, y, h, w, v = self.get_params(image, scale=self.scale, ratio=self.ratio, value=self.value)
            return F.erase(image, x, y, h, w, v, self.inplace)
        return {'image': image,
                'landmarks': landmarks,
                'net': net}



def my_enlargement(img, rect, landmarks):
    angle = random.randint(0, 360)
    M = cv2.getRotationMatrix2D((0, 0), angle, 1)
    dst = cv2.warpAffine(img, M, img.shape[:2])
    rps = [[[max(0, i), max(0, j)] for (i, j) in zip(rect[::2], rect[1::2])]]
    lps = [[max(0, i), max(0, j)] for (i, j) in zip(landmarks[::2], landmarks[1::2])]
    w_rps = cv2.perspectiveTransform(np.float32([rps]), M)
    w_lps = cv2.perspectiveTransform(np.float32([lps]), M)
    return dst, w_rps.flatten(), w_lps.flatten()


class FaceLandmarksDataset(Dataset):
    # Face Landmarks Dataset
    def __init__(self, src_lines, phase, net, roi, angle, transform=None):
        '''
        :param src_lines: src_lines
        :param train: whether we are training or not
        :param transform: data transform
        '''
        self.lines = src_lines
        self.phase = phase
        self.transform = transform
        self.net = net
        self.roi = roi
        self.angle = angle

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        img_name, rect, landmarks = my_parse_line(self.lines[idx])
        # image
        if self.net == '':
            img = Image.open(img_name).convert('L')
        else:
            img = Image.open(img_name).convert('RGB')
        rect = my_expand_roi(rect, img.width, img.height, ratio=self.roi + random.random() * 0.1)
        img_crop = img.crop(tuple(rect))
        landmarks = np.float32([
            landmarks[i] - rect[0] if i % 2 == 0
            else landmarks[i] - rect[1]
            for i in range(len(landmarks))
        ])
        sample = {'image': img_crop, 'landmarks': landmarks, 'net': self.net, 'angle': self.angle}
        sample = self.transform(sample)
        return sample


def load_data(phase, net, roi, angle):
    data_file = phase + '.txt'
    with open(data_file) as f:
        lines = f.readlines()
    if phase == 'Train' or phase == 'train':
        tsfm = transforms.Compose([
            RandomFlip(),
            RandomRotate(),
            Normalize(),  # do channel normalization
            ToTensor(),   # convert to torch type: NxCxHxW
            RandomErasing()
        ]
        )
    else:
        tsfm = transforms.Compose([
            Normalize(),
            ToTensor()
        ])
    data_set = FaceLandmarksDataset(lines, phase, net, roi, angle, transform=tsfm)
    return data_set


# 仅生成文件分类
def my_generate_train_test_list(path, ratio=0.8):
    with open(path + 'label.txt', 'r') as f:
        labels = f.readlines()
    train = random.sample(range(len(labels)), int(len(labels) * ratio))
    with open('train.txt', 'a+') as f:
        for i in train:
            f.write(path + labels[i])
    train_set = set(train)
    test = [i for i in range(len(labels)) if i not in train_set]
    with open('test.txt', 'a+') as f:
        for i in test:
            f.write(path + labels[i])


def get_train_test_set(net, roi, angle):
    train_set = load_data('train', net, roi, angle)
    valid_set = load_data('test', net, roi, angle)
    return train_set, valid_set


if __name__ == '__main__':
    for subfolder in folder_list:
        my_generate_train_test_list(path + '/' + subfolder + '/')
