import csv
import glob
import json
import logging
import os
from enum import Enum
from typing import List, Optional, Union

import tqdm
import numpy as np

from dataclasses import dataclass

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2

import torch
import skimage
from torchvision import transforms, utils
from skimage import io, transform
from skimage.transform import resize as skimage_resize
from skimage import img_as_ubyte


# Adapted from: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return img


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image = sample

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return image


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)


def read_img_from_filename(filename, args=None):
    if args is not None:
        if "detectron2" in args.vision_model:
            img = cv2.imread(filename)

            if img is None:
                try:
                    img = io.imread(filename, )
                except:
                    img = io.imread(filename, plugin="imageio")
                return img
                
            if len(img.shape) < 3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img
    try:
        img = io.imread(filename, )
    except:
        img = io.imread(filename, plugin="imageio")
    return img


def resize_img_skimage(img, new_w, new_h):
    return skimage_resize(img, (new_w, new_h))


def skimage_save(fanme, arr, no_lossy=True):
    if no_lossy:
        arr = img_as_ubyte(arr)
    io.imsave(fanme, arr)


def read_and_transform_img_from_filename(filename, args=None, img_transform_func=None):
    img = read_img_from_filename(filename, args=args)
    if len(img.shape) < 3:  # Grayscale to RGB?
        img = skimage.color.gray2rgb(img)
    if img.shape[-1] > 3:  # Remove the alpha channel.
        img = img[:, :, :3]
    if img_transform_func is not None:
        img = img_transform_func(img)
    return img


if __name__ == "__main__":
    filename = ("data/recipeQA/images/images-qa/test/images-qa"
                "/fudgey-oatmeal-cookie-bars_1_0.jpg")
    img = read_img_from_filename(filename)
    print(img.shape, type(img))

    composed = transforms.Compose([Rescale(256),
                                   RandomCrop(224),
                                   ToTensor()])
    
    img = read_and_transform_img_from_filename(filename,
        img_transform_func=composed)
    print(img.size(), type(img))
