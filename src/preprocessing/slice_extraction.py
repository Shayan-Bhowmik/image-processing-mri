import numpy as np


def extract_valid_slices(volume: np.ndarray, threshold: float = 0.01, fixed_count: int | None = None):
    """
    Extract axial slices and remove near-empty slices.

    Parameters:
        volume (np.ndarray): 3D MRI volume (H, W, D)
        threshold (float): Minimum fraction of non-zero pixels required to keep slice
        fixed_count (int | None): If provided, keep up to this many valid slices
            sampled evenly from the valid-slice sequence.

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

    if fixed_count is None or fixed_count <= 0 or len(valid_slices) <= fixed_count:
        return valid_slices

    sample_positions = np.linspace(0, len(valid_slices) - 1, num=fixed_count)
    sampled = [valid_slices[int(round(pos))] for pos in sample_positions]
    return sampled