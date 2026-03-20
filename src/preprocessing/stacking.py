import numpy as np


def create_25d_samples(slices):
    """
    Ablation: Convert slices into 2D samples (no context).

    Each output sample has shape (1, H, W):
        [current_slice]

    This removes 2.5D context (previous and next slices)
    and keeps only the central slice.
    """
    samples = []
    n = len(slices)

    for i in range(n):
        # Take only the current slice and add channel dimension
        single_slice = np.expand_dims(slices[i], axis=0)  # (1, H, W)
        samples.append(single_slice)

    return samples