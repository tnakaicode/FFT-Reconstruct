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


class ImageRec (object):

    def __init__(self, dirname="./test_png/", name="einstein"):
        self.dir = dirname
        self.bmp = self.dir + name
        self.dat = image.imread(self.bmp + ".bmp")
        self.dat_shp = self.dat.shape

        # padded data
        # (150, 150) -> (450, 450)
        self.dat_pad = np.pad(self.dat, (self.dat_shp, self.dat_shp),
                              'constant', constant_values=((0, 0), (0, 0)))
        self.pad_shp = self.dat_pad.shape
        self.pad_shp_m = (self.pad_shp[0] - 1, self.pad_shp[1] - 1)
        self.pad_shp_p = (self.pad_shp[0] + 2, self.pad_shp[1] + 2)

        # mask
        self.mask = np.ones(self.pad_shp_p)
        self.mask = np.pad(self.mask, (self.pad_shp_m, self.pad_shp_m),
                           'constant', constant_values=((0, 0), (0, 0)))

        # 2D-FFT for padded data
        self.pad_fft = fft.fft2(self.dat_pad)
        self.pad_abs = np.abs(self.pad_fft)

        # Initial Phase in random
        self.phas = np.random.rand(*self.pad_shp) * 2 * np.pi
        self.pad_cmp = self.pad_abs * np.exp(1j * self.phas)

        # number of iterations
        r = 101
        # step size parameter
        beta = 0.8

        self.prev = None
        for s in range(0, r):
            # Generate complex data
            # ampl: original ampli data
            # phas: previous phase data
            self.pad_cmp = self.pad_abs * np.exp(1j * np.angle(self.phas))

            self.pad_inv = fft.ifft2(self.pad_cmp)
            self.pad_inv = np.real(self.pad_inv)
            if self.prev is None:
                self.prev = self.pad_inv.copy()

            # apply real-space constraints
            temp = self.pad_inv
            for (i, j), val in np.ndenumerate(self.pad_inv):
                # image region must be positive
                if self.mask[i, j] == 1 and val < 0:
                    self.pad_inv[i, j] = self.prev[i, j] - beta * val
                # push support region intensity toward zero
                if self.mask[i, j] == 0:
                    self.pad_inv[i, j] = self.prev[i, j] - beta * val
                else:
                    self.pad_inv[i, j] = 0
            
            self.prev = temp
            self.phas = fft.fft2(self.pad_inv)

            # save an image of the progress
            if s % 10 == 0:
                plt.figure()
                plt.imshow(self.prev)
                plt.colorbar()
                plt.savefig(self.dir + str(s) + ".png")
                print(time.ctime(time.time()), s)

            if s % 100 == 0:
                plt.savefig(self.dir + str(s) + ".png")
                np.savetxt(self.dir + str(s) + ".txt", self.prev)


if __name__ == '__main__':
    argvs = sys.argv
    parser = OptionParser()
    parser.add_option("--dir", dest="dir", default="./test_png/")
    parser.add_option("--name", dest="name", default="einstein")
    opt, argc = parser.parse_args(argvs)
    print(argc, opt)

    obj = ImageRec()
