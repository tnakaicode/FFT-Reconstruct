import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import os
import numpy.fft as fft
import scipy.ndimage as nd
import scipy.misc as misc
import cv2
from PIL import Image
from matplotlib import image
from linecache import getline, clearcache
from optparse import OptionParser


if __name__ == '__main__':
    argvs = sys.argv
    parser = OptionParser()
    parser.add_option("--dir", dest="dir", default="./test_png/")
    parser.add_option("--name", dest="name", default="einstein")
    opt, argc = parser.parse_args(argvs)
    print(argc, opt)

    # Read in source image
    bmpfile = opt.dir + opt.name
    source = image.imread(bmpfile + ".bmp")

    # Pad image to simulate oversampling
    pad_len = len(source)
    padded = np.pad(source, ((pad_len, pad_len), (pad_len, pad_len)),
                    'constant', constant_values=((0, 0), (0, 0)))

    print(source.shape)
    print(padded.shape)
    ft = fft.fft2(padded)

    # simulate diffraction pattern
    diffract = np.abs(ft)

    l = len(padded)

    # keep track of where the image is vs the padding
    mask = np.ones((pad_len+2, pad_len+2))
    mask = np.pad(mask, ((pad_len-1, pad_len-1), (pad_len-1, pad_len-1)), 'constant',
                  constant_values=((0, 0), (0, 0)))

    # Initial guess using random phase info
    np.savetxt("mask.txt", mask)
    guess = diffract * np.exp(1j * np.random.rand(l, l) * 2 * np.pi)

    # number of iterations
    r = 1001

    # step size parameter
    beta = 0.8

    # previous result
    prev = None
    for s in range(0, r):
        # apply fourier domain constraints
        update = diffract * np.exp(1j * np.angle(guess))

        inv = fft.ifft2(update)
        inv = np.real(inv)
        if prev is None:
            prev = inv

        # apply real-space constraints
        temp = inv
        inv = prev - beta*inv
        inv *= mask
        """for i in range(0, l):
            for j in range(0, l):
                # image region must be positive
                if inv[i, j] < 0 and mask[i, j] == 1:
                    inv[i, j] = prev[i, j] - beta*inv[i, j]
                # push support region intensity toward zero
                if mask[i, j] == 0:
                    inv[i, j] = prev[i, j] - beta*inv[i, j]"""

        prev = temp

        guess = fft.fft2(inv)

        # save an image of the progress

        if s % 10 == 0:
            plt.figure()
            plt.imshow(prev)
            plt.savefig(opt.dir + str(s)+".png")
            print(s)

        if s % 100 == 0:
            np.savetxt(opt.dir + str(s)+".txt", prev)
