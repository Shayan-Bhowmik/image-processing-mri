import numpy as np
import cv2


def normalize(slice_img):
    """
    Z-score normalization on non-zero voxels
    """

    mask = slice_img != 0

    if mask.sum() == 0:
        return slice_img

    mean = slice_img[mask].mean()
    std = slice_img[mask].std()

    if std == 0:
        std = 1

    slice_img = (slice_img - mean) / std

    return slice_img


def resize_slice(slice_img, size=224):
    """
    Resize slice to CNN input size
    """

    return cv2.resize(slice_img, (size, size))