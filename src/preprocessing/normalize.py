import numpy as np


def zscore_normalize(volume: np.ndarray) -> np.ndarray:
    """
    Perform Z-score normalization on non-zero voxels of an MRI volume.
    """
    mask = volume > 0

    if np.sum(mask) == 0:
        return volume

    mean = volume[mask].mean()
    std = volume[mask].std()

    if std == 0:
        return volume

    volume[mask] = (volume[mask] - mean) / std
    return volume