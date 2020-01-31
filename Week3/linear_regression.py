##
# Data: 2020/1/30 Thurs
# Author: Ruikang Dai
# Description: Kaikeba homework week 3
##
import random
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from matplotlib.widgets import Slider, Button, RadioButtons
import sys


class LINEAR_REGRESSION(object):
    def __init__(self, mn_x=0, mx_x=100, w=random.randint(-100, 100), b=random.randint(-10, 10)):
        self.mn = mn_x
        self.mx = mx_x
        self.w = w
        self.b = b
        print("(W, B): ", self.w, self.b)

    def gen_sample_data(self, num_sample=1000, mn_noise=-1000, mx_noise=1000):
        x_list = []
        y_list = []
        mid = (self.mn + self.mx) / 2
        for i in range(num_sample):
            x = random.randint(self.mn, self.mx) * random.random()
            ctrl = np.cos((x - mid) / 2 / mid * np.pi) if random.random() > 0.1 else 1
            noise = random.randint(mn_noise, mx_noise) * ctrl
            y = self.w * x + self.b + noise
            x_list.append(x)
            y_list.append(y)
        return np.array(x_list), np.array(y_list)

    def inference(self, w, b, x):
        pred_y = w * x + b
        return pred_y

    def eval_loss(self, w, b, x_list, gt_y_list):
        avg_loss = 0
        for i in range(len(x_list)):
            avg_loss += 0.5 * (w * x_list[i] + b - gt_y_list[i]) ** 2
        avg_loss /= len(gt_y_list)
        return avg_loss

    def gradient(self, pred_y, gt_y, x):
        diff = pred_y - gt_y
        dw = diff * x
        db = diff
        return dw, db

    def cal_step_gradient(self, batch_x_list, batch_gt_y_list, w, b, learning_rate):
        avg_dw, avg_db = 0, 0
        batch_size = len(batch_x_list)
        for i in range(batch_size):
            pred_y = self.inference(w, b, batch_x_list[i])
            dw, db = self.gradient(pred_y, batch_gt_y_list[i], batch_x_list[i])
            avg_dw += dw
            avg_db += db
        avg_dw /= batch_size
        avg_db /= batch_size
        w -= learning_rate * avg_dw
        b -= learning_rate * avg_db
        return w, b

    def train(self, x_list, gt_y_list, batch_size=100, lr=0.001, mx_iter=5000, fig=None, show_plt=True):
        w = b = 0
        if not fig: fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x_list, gt_y_list, marker='o', markerfacecolor='none', ls='')
        text_w = plt.text(0.2, 0.95, "w:{0}".format(w), horizontalalignment='center', verticalalignment='center',
                          transform=ax.transAxes)
        text_b = plt.text(0.2, 0.90, "b:{0}".format(b), horizontalalignment='center', verticalalignment='center',
                          transform=ax.transAxes)
        if show_plt: plt.ion()
        for i in range(mx_iter):
            batch_idxs = np.random.choice(len(x_list), batch_size)
            batch_x = [x_list[j] for j in batch_idxs]
            batch_y = [gt_y_list[j] for j in batch_idxs]
            w, b = self.cal_step_gradient(batch_x, batch_y, w, b, lr)
            if show_plt and i % 50 == 0:
                plt.pause(0.1)
                try:
                    ax.lines.remove(lines[0])
                except Exception:
                    pass
                text_w.set_text("w({0}):{1}".format(self.w, w))
                text_b.set_text("b({0}):{1}".format(self.b, b))
                draw_line_x = np.array([self.mn, self.mx])
                lines = ax.plot(draw_line_x, draw_line_x * w + b, color='red')
                plt.show()
                print('w:{0}, b:{1}'.format(w, b))
                print('loss is {}'.format(self.eval_loss(w, b, x_list, gt_y_list)))
        plt.waitforbuttonpress()
        if show_plt: plt.ioff()
        return w, b

if len(sys.argv) < 2:
    axcolor = 'lightgoldenrodyellow'
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.15, bottom=0.25)
    cur_w, cur_b = random.randint(0, 10), random.randint(0, 10)
    linear_reg = LINEAR_REGRESSION(w=cur_w, b=cur_b)
    x, y = linear_reg.gen_sample_data()
    draw_sample, = plt.plot(x, y, marker="o", markerfacecolor='none', ls="")
    ax_w = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=axcolor)
    ax_b = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor=axcolor)
    ax_sp = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)

    slider_w = Slider(ax_w, 'w', -1000, 1000.0, valinit=cur_w)
    slider_b = Slider(ax_b, 'b', -1000, 1000.0, valinit=cur_b)
    slider_sp = Slider(ax_sp, 'samples', 1000, 10000, valinit=1000, valstep=1)


    def update(val):
        w, b, sp = slider_w.val, slider_b.val, int(slider_sp.val)
        linear_reg = LINEAR_REGRESSION(w=w, b=b)
        global x
        global y
        x, y = linear_reg.gen_sample_data(num_sample=sp)
        draw_sample.set_ydata(y)
        draw_sample.set_xdata(x)
        fig.canvas.draw_idle()


    slider_w.on_changed(update)
    slider_b.on_changed(update)
    slider_sp.on_changed(update)
    evaluateax = plt.axes([0.7, 0.001, 0.1, 0.04])
    button = Button(evaluateax, 'Evaluate', color=axcolor, hovercolor='0.975')


    def evaluate(event):
        plt.clf()
        w, b = linear_reg.train(x, y, fig=fig)


    button.on_clicked(evaluate)

    plt.show()

if sys.argv[1] == 'cmd':
    init_lr = {'mn_x': input("Min x[0]:"), 'mx_x': input("Max x[100]:"), 'w': input('w[random.randint(0, 10000)]:'),
               'b': input('b[random.randint(0, 10)]:')}

    init_tr = {'mx_iter': input("Max iteration[1000]:"), 'batch_size': input("Batch size[100]:"),
               'learning_rate': input("Learning rate[0.001]:")}

    init_sp = {'num_sample': input("Random Sampels[1000]:"), 'mn_noise': input("Min noise[-1000]:"),
               'mx_noise': input("Max noise[1000]:")}

    init_lr = {i: eval(init_lr[i]) for i in init_lr if init_lr[i]}
    init_tr = {i: eval(init_tr[i]) for i in init_tr if init_tr[i]}
    init_sp = {i: eval(init_sp[i]) for i in init_sp if init_sp[i]}

    linear_reg = LINEAR_REGRESSION(**init_lr)
    x, y = linear_reg.gen_sample_data(**init_sp)
    w, b = linear_reg.train(x, y, **init_tr)
