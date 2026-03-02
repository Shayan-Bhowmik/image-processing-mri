import numpy as np


def extract_axial_slices(volume):
    
    if volume.ndim != 3:
        raise ValueError(
            f"Expected 3D volume, but got shape {volume.shape}"
        )
    
    slices = []
    depth = volume.shape[2]
    
    for i in range(depth):
        slice_2d = volume[:, :, i]
        slices.append(slice_2d)
    
    return slices


def is_informative_slice(slice_2d, threshold=0.05):
    
    if slice_2d.ndim != 2:
        raise ValueError(
            f"Expected 2D slice, but got shape {slice_2d.shape}"
        )
    
    total_pixels = slice_2d.size
    
    
    non_zero_pixels = np.count_nonzero(slice_2d)
    
    ratio = non_zero_pixels / total_pixels
    
    return ratio >= threshold


def filter_empty_slices(slice_list, threshold=0.05):
    
    filtered = []
    
    for slice_2d in slice_list:
        if is_informative_slice(slice_2d, threshold=threshold):
            filtered.append(slice_2d)
    
    return filtered
