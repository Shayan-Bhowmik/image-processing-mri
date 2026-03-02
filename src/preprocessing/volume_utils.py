import numpy as np
import nibabel as nib


def load_nifti(file_path):
    nifti_img = nib.load(file_path)
    
    
    data = nifti_img.get_fdata()
    
    if data.ndim == 4:
        if data.shape[-1] == 1:
            data = np.squeeze(data, axis=-1)
        else:
            raise ValueError(
                f"4D volume with last dimension {data.shape[-1]} cannot be squeezed to 3D. "
                f"Expected last dimension to be 1."
            )
    
    if data.ndim != 3:
        raise ValueError(
            f"Expected 3D volume after processing, but got shape {data.shape}"
        )
    
    data = data.astype(np.float32)
    
    return data


def zscore_normalize(volume):
   
   
    if volume.ndim != 3:
        raise ValueError(
            f"Expected 3D volume, but got shape {volume.shape}"
        )
    
    
    normalized = volume.copy()

    mask = volume != 0
    if not np.any(mask):
        raise ValueError(
            "Volume contains only zero voxels. Cannot normalize."
        )
    
    mean = np.mean(volume[mask])
    std = np.std(volume[mask])
    
    if std == 0:
        normalized[mask] = 0.0
    else:
        normalized[mask] = (volume[mask] - mean) / std
    
    return normalized
