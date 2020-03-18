##
# Data: 2020/2/27 Tues
# Author: Ruikang Dai
# Description: Kaikeba homework week 5
##
import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# K-means
'''
def init_center(X, centers):
    return random.sample(X, centers)
'''


# K-means++ 实现
def init_center(X, centers):
    assert (centers > 0)
    p = np.ones((len(X)), dtype='float32') / len(X)
    result = []
    for i in range(centers):
        result.append(np.random.choice(range(len(X)), p=p))
        distance = np.square(np.sum(np.array([np.sqrt(np.sum(np.square(X - X[j]), axis=1)) for j in result]), axis=0))
        p = distance / np.sum(distance, axis=0)
    return np.array([X[i] for i in result])


def select_center(X, current_center, ax, cur, color):
    category = [[] for _ in range(len(current_center))]
    distances = np.sqrt(np.sum(np.square(np.expand_dims(X, axis=1) - current_center), axis=2))
    y = []
    for i in range(len(distances)):
        cur_ind = float('inf')
        cur_val = float('inf')
        for j, val in enumerate(distances[i]):
            if cur_val > val:
                cur_val = val
                cur_ind = j
        category[cur_ind].append(X[i])
        y.append(cur_ind)
    cur.remove()
    cur = ax.scatter(X[:, 0], X[:, 1], c=y, marker='.')
    mean = [np.sum(i, axis=0) / len(i) if len(i) > 0 else 0 for i in category]
    return cur, np.array(
        [min(category[i], key=lambda x: np.sqrt(np.sum(np.square(x - mean[i]), axis=0))) if len(category[i]) > 0
         else init_center(X, 1)[0]
         for i in range(len(mean))]
    ), y


ap = argparse.ArgumentParser()
ap.add_argument('-d', '--data_center', required=True,
                help='# of random generated data center')
ap.add_argument('-c', '--center_num', required=True,
                help='# of k-means++ center')
ap.add_argument('-i', '--iteration', required=True,
                help='# of iteration')

args = vars(ap.parse_args())

data_center = eval(args['data_center'])
center_num = eval(args['center_num'])
iteration = eval(args['iteration'])

ax = plt.subplot(1, 1, 1)
plt.ion()
# 生成数据
X, y = make_blobs(n_features=2, centers=data_center)
cur = ax.scatter(X[:, 0], X[:, 1], c=y, marker='.')
# 初始化中心 K-means++方法改进版本，使用距离的平方
centers = init_center(X, center_num)
center_colors = range(center_num)
c_ax = ax.scatter(centers[:, 0], centers[:, 1], c=center_colors, marker='x')
plt.pause(1)
# 记录当前分类信息
prev_y = [-1] * len(X)
for i in range(iteration):
    # K-means选点
    cur, new_centers, next_y = select_center(X, centers, ax, cur, center_colors)
    # 当分类停止变化时，停止循环
    if all([prev_y[i] == next_y[i] for i in range(len(X))]):
        break
    prev_y = next_y
    centers = new_centers
    c_ax.remove()
    c_ax = ax.scatter(centers[:, 0], centers[:, 1], c=center_colors, marker='x')
    plt.pause(0.5)
plt.pause(10)
plt.ioff()
