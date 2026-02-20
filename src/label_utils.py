import os
import numpy as np
import nibabel as nib


def load_segmentation_mask(seg_path: str) -> np.ndarray:
    """
    Loads a segmentation mask from a NIfTI file.

    Parameters:
        seg_path (str): Path to the segmentation .nii file

    Returns:
        np.ndarray: 3D segmentation mask volume
    """
    seg_nii = nib.load(seg_path)              # Load NIfTI file
    seg_volume = seg_nii.get_fdata()          # Convert to numpy array

    return seg_volume


def is_abnormal(seg_volume: np.ndarray) -> int:
    """
    Determines whether tumor exists in the segmentation volume.

    Parameters:
        seg_volume (np.ndarray): 3D segmentation array

    Returns:
        int: 1 if tumor exists, 0 otherwise
    """
    # If any voxel is non-zero → tumor present
    if np.any(seg_volume > 0):
        return 1
    return 0


def get_label_from_patient_folder(patient_folder: str) -> int:
    """
    Automatically finds segmentation file inside a BRATS patient folder
    and returns abnormality label.

    Parameters:
        patient_folder (str): Path to patient directory

    Returns:
        int: 1 (abnormal) or 0 (normal)
    """

    # Find segmentation file
    seg_file = None
    for file in os.listdir(patient_folder):
        if file.endswith("_seg.nii"):
            seg_file = os.path.join(patient_folder, file)
            break

    if seg_file is None:
        raise FileNotFoundError("Segmentation file not found.")

    seg_volume = load_segmentation_mask(seg_file)
    label = is_abnormal(seg_volume)

    return label
    

def get_normal_label_from_oasis(file_path: str) -> int:
    """
    Returns label for OASIS dataset.

    Since OASIS dataset contains healthy controls,
    every MRI volume is labeled as 0 (Normal).

    Parameters:
        file_path (str): Path to OASIS .nii file

    Returns:
        int: 0 (normal)
    """
    return 0