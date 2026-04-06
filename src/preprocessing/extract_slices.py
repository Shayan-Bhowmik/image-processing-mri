import numpy as np


def extract_slices(volume, threshold=0.01):
    """
    Extract useful axial slices from a 3D MRI volume
    """

    slices = []

    depth = volume.shape[2]

    for i in range(depth):

        s = volume[:, :, i]

        # keep slices with enough brain content
        if np.count_nonzero(s) / s.size > threshold:
            slices.append(s)

    return slices