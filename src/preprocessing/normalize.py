import numpy as np


def zscore_normalize(volume: np.ndarray) -> np.ndarray:
    """
    Perform Z-score normalization on non-zero voxels of an MRI volume.

    Parameters:
        volume (np.ndarray): 3D MRI volume

    Returns:
        np.ndarray: Normalized volume
    """
    mask = volume > 0

    # If no non-zero voxels exist, return original volume
    if np.sum(mask) == 0:
        return volume

    mean = volume[mask].mean()
    std = volume[mask].std()

    # Prevent division by zero
    if std == 0:
        return volume

    volume[mask] = (volume[mask] - mean) / std
    return volume