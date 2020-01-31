##
# Data: 2020/1/31 Fri
# Author: Ruikang Dai
# Description: Kaikeba homework week 3
##
import random
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from scipy.optimize import fmin_tnc
from matplotlib.widgets import Slider, Button, RadioButtons


class LOGISTIC_REGRESSION(object):
    def __init__(self, mn_x=0, mx_x=100):
        self.mn = mn_x
        self.mx = mx_x

    def dist(self, point_a, point_b):
        return ((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2) ** 0.5

    def gen_sample_points(self, center, n, d, chance=0.95):
        points = []
        for i in range(n):
            x = random.randint(-self.mx * 2, self.mx * 2)
            y = random.randint(-self.mx * 2, self.mx * 2)
            while self.dist(center, (x, y)) > d / 1.8 and (random.random() < chance or self.dist(center, (x, y)) > d):
                x = random.randint(-self.mx * 2, self.mx * 2)
                y = random.randint(-self.mx * 2, self.mx * 2)
            points.append((x, y))
        return points

    def gen_sample_data(self, sample_a_n=500, sample_b_n=500, center_entropy=120):
        center_a = random.randint(self.mn, self.mx), random.randint(self.mn, self.mx)
        center_b = random.randint(self.mn, self.mx), random.randint(self.mn, self.mx)
        while self.dist(center_a, center_b) < center_entropy:
            center_a = random.randint(self.mn, self.mx), random.randint(self.mn, self.mx)
            center_b = random.randint(self.mn, self.mx), random.randint(self.mn, self.mx)
        self.center_a, self.center_b = center_a, center_b
        return np.array(self.gen_sample_points(center_a, sample_a_n, self.dist(center_a, center_b))), \
               np.array(self.gen_sample_points(center_b, sample_b_n, self.dist(center_a, center_b)))

    def sigmoid(self, x):
        # Activation function used to map any real value between 0 and 1
        return 1 / (1 + np.exp(-x))

    def net_input(self, theta, x):
        # Computes the weighted sum of inputs
        return np.dot(x, theta)

    def probability(self, theta, x):
        # Returns the probability after passing through sigmoid
        return self.sigmoid(self.net_input(theta, x))

    def cost_function(self, theta, x, y):
        # Computes the cost function for all the training samples
        m = x.shape[0]
        total_cost = -(1 / m) * np.sum(
            y * np.log(self.probability(theta, x)) + (1 - y) * np.log(
                1 - self.probability(theta, x)))
        return total_cost

    def gradient(self, theta, x, y):
        # Computes the gradient of the cost function at the point theta
        m = x.shape[0]
        return (1 / m) * np.dot(x.T, self.sigmoid(self.net_input(theta, x)) - y)

    def fit(self, X, y, theta):
        opt_weights = fmin_tnc(func=self.cost_function, x0=theta, fprime=self.gradient, args=(X, y.flatten()))
        return opt_weights[0]

    def train(self, point_a, point_b, batch_size=100, lr=0.001, mx_iter=1000, fig=None, show_plt=True):
        if not fig: fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        self.plot = fig
        self.ax = ax
        ax.plot(point_a[::, 0], point_a[::, 1], color="blue", marker="o", markerfacecolor='none', ls="")
        ax.plot(point_b[::, 0], point_b[::, 1], color="red", marker="o", markerfacecolor='none', ls="")
        cur_mn_y, cur_mx_y = min(min(point_a[::, 1]), min(point_b[::, 1])), max(max(point_a[::, 1]),
                                                                                max(point_b[::, 1]))
        ax.set_ylim([cur_mn_y + 10, cur_mx_y + 10])
        if show_plt: plt.ion()
        for i in range(mx_iter):
            batch_a_idx = np.random.choice(len(point_a), batch_size)
            batch_b_idx = np.random.choice(len(point_b), batch_size)
            raw_x, raw_y = [], []
            for j in range(batch_size):
                if random.random() > 0.5:
                    raw_x.append(point_a[batch_a_idx[j]])
                    raw_y.append(0)
                else:
                    raw_x.append(point_b[batch_b_idx[j]])
                    raw_y.append(1)
            X = np.array(raw_x)
            y = np.array(raw_y)
            X = np.c_[np.ones(X.shape[0]), X]
            y = y[:, np.newaxis]
            try:
                thetas
            except NameError:
                thetas = np.zeros((X.shape[1], 1))
            thetas = self.fit(X, y, thetas)
            if show_plt and i % 10 == 0:
                plt.pause(0.1)
                try:
                    ax.lines.remove(lines[0])
                except Exception:
                    pass
                x_values = [np.min(X[:, 1] - 5), np.max(X[:, 2] + 5)]
                lines = ax.plot(x_values, - (thetas[0] + np.dot(thetas[1], x_values)) / thetas[2], color='green')
                plt.show()
        self.thetas = thetas
        self.test()
        if show_plt: plt.ioff()

    def predict(self, x):
        theta = self.thetas[:, np.newaxis]
        return self.probability(theta, x)

    def accuracy(self, x, actual_classes, probab_threshold=0.5):
        predicted_classes = (self.predict(x) >=
                             probab_threshold).astype(int)
        predicted_classes = predicted_classes.flatten()
        accuracy = np.mean(predicted_classes == actual_classes)
        return accuracy * 100

    def test(self):
        point_a = np.array(self.gen_sample_points(self.center_a, 50, 120, chance=1))
        point_b = np.array(self.gen_sample_points(self.center_b, 50, 120, chance=1))
        self.ax.scatter(point_a[:,0], point_a[::,1], color='blue')
        self.ax.scatter(point_b[:,0], point_b[::,1], color='red')
        X = np.c_[np.ones(100), np.r_[point_a, point_b]]
        y = np.r_[np.zeros(50), np.ones(50)]
        y = y[:, np.newaxis]
        self.ax.text(0.2, 0.95, "New prediction accuracy:{0}".format(self.accuracy(X, y.flatten())), transform=ax.transAxes)



axcolor = 'lightgoldenrodyellow'
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.15, bottom=0.25)
cur_a, cur_b, d = random.randint(20, 100), random.randint(20, 100), random.randint(50, 130)
mn, mx = 0, 100
if mx - mn < d:
    mx += d // 2
    mn -= d // 2
logistic_reg = LOGISTIC_REGRESSION(mn_x=mn, mx_x=mx)
a, b = logistic_reg.gen_sample_data(sample_a_n=cur_a, sample_b_n=cur_b, center_entropy=d)
draw_sample1, = plt.plot(a[::, 0], a[::, 1], color="blue", marker="o", markerfacecolor='none', ls="")
draw_sample2, = plt.plot(b[::, 0], b[::, 1], color="red", marker="o", markerfacecolor='none', ls="")
ax_w = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=axcolor)
ax_b = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor=axcolor)
ax_sp = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)

slider_d = Slider(ax_w, 'distance', 1, 130, valinit=d)
slider_spa = Slider(ax_b, 'sample a', 10, 500, valinit=cur_a, valstep=10)
slider_spb = Slider(ax_sp, 'sample b', 10, 500, valinit=cur_b, valstep=10)


def update(val):
    d, spa, spb = slider_d.val, int(slider_spa.val), int(slider_spb.val)
    global logistic_reg
    logistic_reg = LOGISTIC_REGRESSION(mn_x=mn, mx_x=mx)
    global a
    global b
    a, b = logistic_reg.gen_sample_data(sample_a_n=spa, sample_b_n=spb, center_entropy=d)
    draw_sample1.set_ydata(a[::, 0])
    draw_sample1.set_xdata(a[::, 1])
    draw_sample2.set_ydata(b[::, 0])
    draw_sample2.set_xdata(b[::, 1])
    fig.canvas.draw_idle()


slider_d.on_changed(update)
slider_spa.on_changed(update)
slider_spb.on_changed(update)
evaluateax = plt.axes([0.7, 0.001, 0.1, 0.04])
button = Button(evaluateax, 'Evaluate', color=axcolor, hovercolor='0.975')


def evaluate(event):
    plt.clf()
    global a, b
    logistic_reg.train(a, b, fig=fig)


button.on_clicked(evaluate)

plt.show()
