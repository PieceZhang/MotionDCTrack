# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2

from siamban.utils.bbox import corner2center, \
        Center, center2corner, Corner
from scipy import interpolate


class Augmentation:
    def __init__(self, shift, scale, blur, flip, color):
        self.shift = shift
        self.scale = scale
        self.blur = blur
        self.flip = flip
        self.color = color
        self.rgbVar = np.array(
            [[-0.55919361,  0.98062831, - 0.41940627],
             [1.72091413,  0.19879334, - 1.82968581],
             [4.64467907,  4.73710203, 4.88324118]], dtype=np.float32)

    @staticmethod
    def random():
        return np.random.random() * 2 - 1.0

    def _crop_roi(self, image, bbox, out_sz, padding=(0, 0, 0)):
        bbox = [float(x) for x in bbox]
        a = (out_sz-1) / (bbox[2]-bbox[0])
        b = (out_sz-1) / (bbox[3]-bbox[1])
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float32)
        crop = cv2.warpAffine(image, mapping, (out_sz, out_sz),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=padding)
        return crop

    def _blur_aug(self, image):
        def rand_kernel():
            sizes = np.arange(5, 46, 2)
            size = np.random.choice(sizes)
            kernel = np.zeros((size, size))
            c = int(size/2)
            wx = np.random.random()
            kernel[:, c] += 1. / size * wx
            kernel[c, :] += 1. / size * (1-wx)
            return kernel
        kernel = rand_kernel()
        image = cv2.filter2D(image, -1, kernel)
        return image

    def _color_aug(self, image):
        offset = np.dot(self.rgbVar, np.random.randn(3, 1))
        offset = offset[::-1]  # bgr 2 rgb
        offset = offset.reshape(3)
        image = image - offset
        return image

    def _gray_aug(self, image):
        grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(grayed, cv2.COLOR_GRAY2BGR)
        return image

    def _shift_scale_aug(self, image, bbox, crop_bbox, size, neg=False, histm=None, start_frame=False):
        im_h, im_w = image.shape[:2]

        # adjust crop bounding box
        crop_bbox_center = corner2center(crop_bbox)
        if self.scale:
            scale_x = (1.0 + Augmentation.random() * self.scale)
            scale_y = (1.0 + Augmentation.random() * self.scale)
            h, w = crop_bbox_center.h, crop_bbox_center.w
            scale_x = min(scale_x, float(im_w) / w)
            scale_y = min(scale_y, float(im_h) / h)
            crop_bbox_center = Center(crop_bbox_center.x,
                                      crop_bbox_center.y,
                                      crop_bbox_center.w * scale_x,
                                      crop_bbox_center.h * scale_y)

        crop_bbox = center2corner(crop_bbox_center)
        if self.shift:
            sx = Augmentation.random() * self.shift
            sy = Augmentation.random() * self.shift

            x1, y1, x2, y2 = crop_bbox

            sx = max(-x1, min(im_w - 1 - x2, sx))
            sy = max(-y1, min(im_h - 1 - y2, sy))

            crop_bbox = Corner(x1 + sx, y1 + sy, x2 + sx, y2 + sy)

        # adjust target bounding box
        x1, y1 = crop_bbox.x1, crop_bbox.y1
        bbox = Corner(bbox.x1 - x1, bbox.y1 - y1,
                      bbox.x2 - x1, bbox.y2 - y1)

        if self.scale:
            bbox = Corner(bbox.x1 / scale_x, bbox.y1 / scale_y,
                          bbox.x2 / scale_x, bbox.y2 / scale_y)

        image = self._crop_roi(image, crop_bbox, size)
        return image, bbox

    def _flip_aug(self, image, bbox):
        image = cv2.flip(image, 1)
        width = image.shape[1]
        bbox = Corner(width - 1 - bbox.x2, bbox.y1,
                      width - 1 - bbox.x1, bbox.y2)
        return image, bbox

    def __call__(self, image, bbox, size, gray=False, neg=False, histm=None, start_frame=False):
        if isinstance(image, list) and isinstance(bbox, list):
            shape = image[0].shape
        else:
            shape = image.shape
        crop_bbox = center2corner(Center(shape[0]//2, shape[1]//2,
                                         size-1, size-1))  # size: 127*127 or 255*255
        # gray augmentation
        if gray:
            image = self._gray_aug(image)

        # shift scale augmentation
        image, bbox = self._shift_scale_aug(image, bbox, crop_bbox, size, neg, histm, start_frame)

        # color augmentation
        if self.color > np.random.random():
            image = self._color_aug(image)

        # blur augmentation
        if self.blur > np.random.random():
            image = self._blur_aug(image)

        # flip augmentation
        if self.flip and self.flip > np.random.random():
            image, bbox = self._flip_aug(image, bbox)
        return image, bbox


class Augmentation2(Augmentation):
    """
    Do augmentation for multiple images, and use the same random params
    """
    def __init__(self, shift, scale, blur, flip, color):
        super().__init__(shift, scale, blur, flip, color)

    def _gray_aug(self, imagelist):
        outimage = []
        for image in imagelist:
            grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(grayed, cv2.COLOR_GRAY2BGR)
            outimage.append(image)
        return outimage

    def _shift_scale_aug(self, imagelist, bboxlist, crop_bbox_original, size, neg=False, histm=None, start_frame=False):
        outimage = []
        outbox = []
        randparam = [Augmentation.random(), Augmentation.random(),
                     Augmentation.random(), Augmentation.random()]
        if start_frame:  # use smaller range for starter frame
            randparam = list(map(lambda x: x / 5, randparam))
        if neg:
            for i in range(2, len(randparam)):
                if 0 <= randparam[i] < 0.5:
                    randparam[i] += 0.5
                elif -0.5 < randparam[i] < 0:
                    randparam[i] -= 0.5
        for image, bbox in zip(imagelist, bboxlist):
            im_h, im_w = image.shape[:2]
            crop_bbox = crop_bbox_original
            # adjust crop bounding box
            crop_bbox_center = corner2center(crop_bbox)  # size: 127*127 or 255*255, e.g. Center(x=255.0, y=255.0, w=254.0, h=254.0)
            if self.scale:
                scale_x = (1.0 + randparam[0] * self.scale)
                scale_y = (1.0 + randparam[1] * self.scale)
                h, w = crop_bbox_center.h, crop_bbox_center.w
                scale_x = min(scale_x, float(im_w) / w)
                scale_y = min(scale_y, float(im_h) / h)
                crop_bbox_center = Center(crop_bbox_center.x,
                                          crop_bbox_center.y,
                                          crop_bbox_center.w * scale_x,
                                          crop_bbox_center.h * scale_y)  # rescale cropbbox w h, e.g. Center(x=255.0, y=255.0, w=231.91, h=284.18)

            crop_bbox = center2corner(crop_bbox_center)
            if self.shift:
                sx = randparam[2] * self.shift
                sy = randparam[3] * self.shift

                if histm is not None:
                    lenm = histm.shape[0] // 3
                    fx = interpolate.UnivariateSpline(np.linspace(0, lenm-1, lenm), histm[-lenm:, 0])
                    fy = interpolate.UnivariateSpline(np.linspace(0, lenm-1, lenm), histm[-lenm:, 1])
                    sx_pred = fx(lenm)
                    sy_pred = fy(lenm)
                    sx -= sx_pred
                    sy -= sy_pred

                x1, y1, x2, y2 = crop_bbox

                sx = max(-x1, min(im_w - 1 - x2, sx))
                sy = max(-y1, min(im_h - 1 - y2, sy))

                crop_bbox = Corner(x1 + sx, y1 + sy, x2 + sx, y2 + sy)

            # adjust target bounding box: bbox based on original image, crop_bbox based on cropped subwindow
            x1, y1 = crop_bbox.x1, crop_bbox.y1  # left top corner of crop_bbox
            bbox = Corner(bbox.x1 - x1, bbox.y1 - y1,
                          bbox.x2 - x1, bbox.y2 - y1)  # translate bbox (image coordinate) to crop_bbox (subwindow coordinate)

            if self.scale:
                bbox = Corner(bbox.x1 / scale_x, bbox.y1 / scale_y,
                              bbox.x2 / scale_x, bbox.y2 / scale_y)

            image = self._crop_roi(image, crop_bbox, size)
            outimage.append(image)
            outbox.append(bbox)

        return outimage, outbox

    def _color_aug(self, imagelist):
        outimage = []
        randparam = np.random.randn(3, 1)
        for image in imagelist:
            offset = np.dot(self.rgbVar, randparam)
            offset = offset[::-1]  # bgr 2 rgb
            offset = offset.reshape(3)
            image = image - offset
            outimage.append(image)
        return outimage

    def _blur_aug(self, imagelist):
        def rand_kernel():
            sizes = np.arange(5, 46, 2)
            size = np.random.choice(sizes)
            kernel = np.zeros((size, size))
            c = int(size/2)
            wx = np.random.random()
            kernel[:, c] += 1. / size * wx
            kernel[c, :] += 1. / size * (1-wx)
            return kernel
        kernel = rand_kernel()
        outimage = []
        for image in imagelist:
            image = cv2.filter2D(image, -1, kernel)
            outimage.append(image)
        return outimage

    def _flip_aug(self, imagelist, bboxlist):
        outimage = []
        outbox = []
        for image, bbox in zip(imagelist, bboxlist):
            image = cv2.flip(image, 1)
            width = image.shape[1]
            bbox = Corner(width - 1 - bbox.x2, bbox.y1,
                          width - 1 - bbox.x1, bbox.y2)
            outimage.append(image)
            outbox.append(bbox)
        return outimage, outbox

