import nibabel as nib
import numpy as np


def load_nifti(path: str) -> np.ndarray:
    """
    Load a NIfTI file and return as float32 numpy array.
    """
    nii = nib.load(path)
    volume = nii.get_fdata()
    return volume.astype(np.float32)