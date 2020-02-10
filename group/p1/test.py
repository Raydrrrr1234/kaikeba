import cv2
import numpy as np
from scipy.ndimage.filters import convolve

def gaussian_filter(sigma):
    size = 2*np.ceil(3*sigma)+1
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(sigma**2))) / (2*np.pi*sigma**2)
    return g/g.sum()

my_gaussian = gaussian_filter(0.5)

def generate_octave(init_level, s, sigma):
    octave = [init_level]
    k = 2**(1/s)
    kernel = gaussian_filter(k * sigma)
    for _ in range(s+2):
        next_level = convolve(octave[-1], kernel)
        octave.append(next_level)
    return octave