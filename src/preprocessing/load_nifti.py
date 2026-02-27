import nibabel as nib
import numpy as np


def load_nifti(path: str) -> np.ndarray:
    """
    Load a NIfTI file directly as float32 to reduce memory usage.
    """
    nii = nib.load(path)

    # Load directly as float32 (prevents float64 allocation)
    volume = nii.get_fdata(dtype=np.float32)

    return volume