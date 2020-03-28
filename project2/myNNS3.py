import warnings
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit.annotations import Optional, Tuple
from torch import Tensor


# Default NN from project Demo
class Net(nn.Module):
    def __init__(self, use_bn):
        super(Net, self).__init__()
        # Backbone:
        # in_channel, out_channel, kernel_size, stride, padding
        # block 1
        self.conv1_1 = nn.Conv2d(1, 8, 5, 2, 0)
        if use_bn: self.bn1 = nn.BatchNorm2d(8)
        # block 2
        self.conv2_1 = nn.Conv2d(8, 16, 3, 1, 0)
        self.conv2_2 = nn.Conv2d(16, 16, 3, 1, 0)
        # block 3
        self.conv3_1 = nn.Conv2d(16, 24, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(24, 24, 3, 1, 0)
        # block 4
        self.conv4_1 = nn.Conv2d(24, 40, 3, 1, 1)
        # points branch
        self.conv4_2 = nn.Conv2d(40, 80, 3, 1, 1)
        self.ip1 = nn.Linear(4 * 4 * 80, 128)
        self.ip2 = nn.Linear(128, 128)
        self.ip3 = nn.Linear(128, 42)
        # common used
        self.prelu1_1 = nn.PReLU()
        self.prelu2_1 = nn.PReLU()
        self.prelu2_2 = nn.PReLU()
        self.prelu3_1 = nn.PReLU()
        self.prelu3_2 = nn.PReLU()
        self.prelu4_1 = nn.PReLU()
        self.prelu4_2 = nn.PReLU()
        self.preluip1 = nn.PReLU()
        self.preluip2 = nn.PReLU()
        self.ave_pool = nn.AvgPool2d(2, 2, ceil_mode=True)

    def forward(self, x):
        # block 1
        # print('x input shape: ', x.shape)
        x = self.conv1_1(x)
        try:
            x = self.bn1(x)
        except AttributeError:
            pass
        x = self.ave_pool(self.prelu1_1(x))
        # print('x after block1 and pool shape should be nx8x27x27: ', x.shape)     # good
        # block 2
        x = self.prelu2_1(self.conv2_1(x))
        # print('b2: after conv2_1 and prelu shape should be nx16x25x25: ', x.shape) # good
        x = self.prelu2_2(self.conv2_2(x))
        # print('b2: after conv2_2 and prelu shape should be nx16x23x23: ', x.shape) # good
        x = self.ave_pool(x)
        # print('x after block2 and pool shape should be nx16x12x12: ', x.shape)
        # block 3
        x = self.prelu3_1(self.conv3_1(x))
        # print('b3: after conv3_1 and pool shape should be nx24x10x10: ', x.shape)
        x = self.prelu3_2(self.conv3_2(x))
        # print('b3: after conv3_2 and pool shape should be nx24x8x8: ', x.shape)
        x = self.ave_pool(x)
        # print('x after block3 and pool shape should be nx24x4x4: ', x.shape)
        # block 4
        x = self.prelu4_1(self.conv4_1(x))
        # print('x after conv4_1 and pool shape should be nx40x4x4: ', x.shape)

        # points branch
        ip3 = self.prelu4_2(self.conv4_2(x))
        # print('pts: ip3 after conv4_2 and pool shape should be nx80x4x4: ', ip3.shape)
        ip3 = ip3.view(-1, 4 * 4 * 80)
        # print('ip3 flatten shape should be nx1280: ', ip3.shape)
        ip3 = self.preluip1(self.ip1(ip3))
        # print('ip3 after ip1 shape should be nx128: ', ip3.shape)
        ip3 = self.preluip2(self.ip2(ip3))
        # print('ip3 after ip2 shape should be nx128: ', ip3.shape)
        ip3 = self.ip3(ip3)
        # print('ip3 after ip3 shape should be nx42: ', ip3.shape)

        return ip3
