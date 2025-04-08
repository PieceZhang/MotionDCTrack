from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import numpy as np
from siamban.core.config import cfg


def motion_normalize(histm: list or np.array):  # deltax, deltay, w, h
    if isinstance(histm, list):
        histm = np.array(histm)
        if len(histm.shape) == 1:
            histm = histm[None, :]
    histm = histm.astype(np.float32)
    # assuming delta ranges from min1 to max1
    min1 = -30
    max1 = 30
    histm[:, :2] = (histm[:, :2] - min1) / (max1 - min1)
    histm[:, :2] = np.clip(histm[:, :2], 0, 1)
    # assuming w, h ranges from min2 to max2
    min2 = 10
    max2 = 100
    histm[:, 2:] = (histm[:, 2:] - min2) / (max2 - min2)
    histm[:, 2:] = np.clip(histm[:, 2:], 0, 1)
    return histm


class MotionBuffer(object):
    def __init__(self, n, s, L, Gauss_mu, Gauss_sigma, **kwargs):
        assert n*s == L, "n * s should be equal to L"
        self.n = n
        self.s = s
        self.L = L
        self.mu = Gauss_mu
        self.sigma = Gauss_sigma
        self.maxtime = n // 3
        self.maxlength = self.maxtime * 30
        self.buffer = []  # L * 4, (delta_x, delta_y, w, h)
        self.sampling_locater = np.linspace(0, L-1, L) * (30/(s*3))
        if s * 3 < 30:
            self.sampling_locater += 1
        self.sampling_locater = self.sampling_locater.astype(np.int32)

    # def _gen_noise(self):
    #     return np.concatenate([np.random.normal(self.mu, self.sigma / 10, 2), np.random.normal(self.mu, self.sigma, 2)])

    def init(self, w, h):
        # self.buffer = [motion_normalize(np.array([0, 0, w, h])[None, :] + self._gen_noise())
        #                for _ in range(self.maxlength)]
        self.buffer = [motion_normalize([0, 0, w, h]) for _ in range(self.maxlength)]

    def get(self):
        return np.array(self.buffer, dtype=np.float32)[self.sampling_locater].transpose(1, 0, 2)

    def pop(self, index=0):
        return self.buffer.pop(index)

    def append(self, item):  # [delta_x, delta_y, width, height]
        if len(self.buffer) >= self.maxlength:
            self.pop()
        item = motion_normalize(item)
        self.buffer.append(item)


if __name__ == '__main__':
    # for debug
    mb = MotionBuffer(6, 5, 30, 0, 1)
    mb.init(50, 50)
    mb.append([10, -10, 50, 50])
    get = mb.get()
    print(get)
    pass
