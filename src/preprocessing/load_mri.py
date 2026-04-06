import nibabel as nib
import numpy as np


def load_nifti(file_path):
    """
    Load a NIfTI MRI file and return the volume as numpy array
    """

    mri = nib.load(file_path)

    volume = mri.get_fdata()

    volume = np.asarray(volume, dtype=np.float32)

    if volume.ndim == 4:
        volume = volume[:, :, :, 0]

    return volume