#!/usr/bin/python3
# coding=utf-8

import cv2
import paddle
import numpy as np
from PIL import ImageEnhance
import random
import paddle
import paddle
import random
from paddle.nn import functional as F


class Compose(object):
    def __init__(self, *ops):
        self.ops = ops

    def __call__(self, image, mask):
        for op in self.ops:
            image, mask = op(image, mask)
        return image, mask


class RandomVorizontalFlip(object):
    def __call__(self, image, mask=None):
        if np.random.randint(5) == 1:
            image = image[::-1, :, :].copy()
            if mask is not None:
                mask = mask[::-1, :, :].copy()
        return image.copy(), mask


class RandomHorizontalFlip(object):
    def __call__(self, image, mask=None):
        if np.random.randint(2) == 1:
            image = image[:, ::-1, :].copy()
            if mask is not None:
                mask = mask[:, ::-1, :].copy()
        return image, mask


class Normalize(object):
    def __init__(self):
        self.mean = np.array([[[124.55, 118.90, 102.94]]])
        self.std  = np.array([[[56.77, 55.97, 57.50]]])

    def __call__(self, image, mask=None):
        image = (image - self.mean) / self.std
        if mask is not None:
            mask /= 255
            return image, mask
        else:
            return image, mask


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask=None):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
            return image, mask
        else:
            return image, mask


class RandomBrightness(object):
    def __call__(self, image, mask=None):
        contrast = np.random.rand(1) + 0.5
        light = np.random.randint(-15, 15)
        inp_img = contrast * image + light
        return np.clip(inp_img, 0, 255), mask


class RandomCrop(object):
    def __call__(self, image, mask=None):
        H, W, _ = image.shape
        randw = np.random.randint(W / 8)
        randh = np.random.randint(H / 8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw
        image = image[p0:p1, p2:p3, :]
        if mask is not None:
            mask = mask[p0:p1, p2:p3, :]
        return image, mask


class RandomBlur:
    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, im, label=None):
        if self.prob <= 0:
            n = 0
        elif self.prob >= 1:
            n = 1
        else:
            n = int(1.0 / self.prob)
        if n > 0:
            if np.random.randint(0, n) == 0:
                radius = np.random.randint(3, 10)
                if radius % 2 != 1:
                    radius = radius + 1
                if radius > 9:
                    radius = 9
                im = cv2.GaussianBlur(im, (radius, radius), 0, 0)
        return im, label


class ToTensor(object):
    def __call__(self, image, mask=None):
        image = image.transpose((2, 0, 1))
        if mask is not None:
            mask = mask.transpose((2, 0, 1))
            mask = mask.mean(axis=0, keepdims=True)
            image, mask = image.astype(np.float32), mask.astype(np.float32)
            return image, mask
        else:
            return image.astype(np.float32), mask





