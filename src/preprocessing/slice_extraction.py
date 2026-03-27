import numpy as np


def extract_valid_slices(volume: np.ndarray, threshold: float = 0.01):
    """
    Extract axial slices and remove near-empty slices.

    Parameters:
        volume (np.ndarray): 3D MRI volume (H, W, D)
        threshold (float): Minimum fraction of non-zero pixels required to keep slice

    Returns:
        list[np.ndarray]: List of valid 2D slices
    """
    valid_slices = []

    height, width, depth = volume.shape
    total_pixels = height * width

    for i in range(depth):
        slice_ = volume[:, :, i]

        non_zero_ratio = np.count_nonzero(slice_) / total_pixels

        if non_zero_ratio > threshold:
            valid_slices.append(slice_)

    return valid_slices