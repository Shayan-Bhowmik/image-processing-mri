import numpy as np


def zscore_normalize(volume: np.ndarray) -> np.ndarray:
    """
    Ablation 3: Normalization removed.

    This function now returns the raw MRI volume without any scaling.
    Only ensures the data type is float32 for consistency.
    """
    return volume.astype(np.float32)