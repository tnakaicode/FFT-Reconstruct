import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import os
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
        self.dat_shp_m = (self.dat_shp[0] - 1, self.dat_shp[1] - 1)
        self.dat_shp_p = (self.dat_shp[0] + 2, self.dat_shp[1] + 2)

        self.pad_data()

        # 2D-FFT for padded data
        # Initialize
        self.pad_fft = np.fft.fft2(self.dat_pad)
        self.pad_abs = np.abs(self.pad_fft)
        self.phs = np.random.rand(*self.pad_shp) * 2 * np.pi
        self.pad_cmp = self.pad_abs * np.exp(1j * self.phs)

    def pad_data(self):
        # padded data
        # (150, 150) -> (450, 450)
        self.dat_pad = np.pad(
            self.dat, (self.dat_shp, self.dat_shp), 'constant')
        self.pad_shp = self.dat_pad.shape

        plt.figure()
        plt.imshow(self.dat_pad)
        plt.colorbar()
        plt.savefig(self.bmp + ".png")

        # mask
        self.mask = np.ones(self.dat_shp_p)
        self.mask = np.pad(
            self.mask, (self.dat_shp_m, self.dat_shp_m), 'constant')

    def run_fft(self, r=1001, beta=0.9):
        self.prev = np.real(self.pad_inv)
        self.pad_cmp = self.pad_abs * np.exp(1j * np.angle(self.pad_cmp))
        self.pad_inv = np.fft.ifft2(self.pad_cmp)
        self.prv = np.real(self.pad_inv)
        for s in range(0, r):
            # Generate complex data
            # ampl: original ampli data
            # phas: previous phase data
            self.pad_cmp = self.pad_abs * np.exp(1j * np.angle(self.pad_cmp))
            self.pad_inv = np.fft.ifft2(self.pad_cmp)
            self.pad_rel = np.real(self.pad_inv)

            # apply real-space constraints
            for (i, j), val in np.ndenumerate(self.pad_rel):
                # image region must be positive
                if self.mask[i, j] == 1 and val < 0:
                    self.pad_rel[i, j] = self.prv[i, j] - beta * val
                # push support region intensity toward zero
                if self.mask[i, j] == 0:
                    self.pad_rel[i, j] = self.prv[i, j] - beta * val

            self.prv = np.real(self.pad_inv)
            self.pad_cmp = np.fft.fft2(self.pad_rel)

            # save an image of the progress
            if s % 10 == 0:
                plt.figure()
                plt.imshow(self.prv)
                plt.colorbar()
                plt.savefig(self.dir + "{0:04d}.png".format(s))
                print(time.ctime(time.time()), s)

            if s % 100 == 0:
                np.savetxt(self.dir + str(s) + ".txt", self.prv)


if __name__ == '__main__':
    argvs = sys.argv
    parser = OptionParser()
    parser.add_option("--dir", dest="dir", default="./test_png/")
    parser.add_option("--name", dest="name", default="einstein")
    opt, argc = parser.parse_args(argvs)
    print(argc, opt)

    obj = ImageRec()
    obj.run_fft(r=101, beta=0.8)
    obj.run_fft(r=101, beta=0.9)
    obj.run_fft(r=101, beta=1.0)
