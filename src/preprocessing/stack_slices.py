import numpy as np


def stack_slices(slices):
    """
    Create 2.5D input by stacking previous, current, next slice
    """

    stacked = []

    for i in range(len(slices)):

        prev_slice = slices[i - 1] if i > 0 else slices[i]
        curr_slice = slices[i]
        next_slice = slices[i + 1] if i < len(slices) - 1 else slices[i]

        stacked_slice = np.stack([prev_slice, curr_slice, next_slice])

        stacked.append(stacked_slice)

    return stacked