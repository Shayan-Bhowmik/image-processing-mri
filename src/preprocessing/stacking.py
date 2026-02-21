import numpy as np


def create_25d_samples(slices):
    """
    Convert a list of 2D slices into 2.5D stacked samples.

    Each output sample has shape (3, H, W):
        [previous_slice, current_slice, next_slice]

    Edge slices are duplicated to maintain consistency.
    """
    samples = []
    n = len(slices)

    for i in range(n):
        prev_slice = slices[i - 1] if i > 0 else slices[i]
        next_slice = slices[i + 1] if i < n - 1 else slices[i]

        stacked = np.stack([prev_slice, slices[i], next_slice], axis=0)
        samples.append(stacked)

    return samples