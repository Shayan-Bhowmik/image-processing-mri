import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


class MRIDataset(Dataset):

    def __init__(self, file_paths, label):
        self.file_paths = file_paths
        self.label = label

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):

        path = self.file_paths[idx]

        try:
            # Load MRI volume
            img = nib.load(path)
            volume = img.get_fdata()

            # Safety check
            if volume.ndim != 3:
                raise ValueError("Invalid MRI shape")

            # Safe center slice
            slice_index = volume.shape[2] // 2
            slice_index = max(1, min(slice_index, volume.shape[2] - 2))

            prev_slice = volume[:, :, slice_index - 1]
            curr_slice = volume[:, :, slice_index]
            next_slice = volume[:, :, slice_index + 1]

            # Stack slices (H,W,3)
            slice_25d = np.stack([prev_slice, curr_slice, next_slice], axis=-1)

            # Remove NaN / Inf
            slice_25d = np.nan_to_num(slice_25d)

            # Convert dtype
            slice_25d = slice_25d.astype(np.float32)

            # Check empty slice
            if slice_25d.size == 0:
                raise ValueError("Empty slice")

            # Resize
            slice_25d = cv2.resize(slice_25d, (224, 224), interpolation=cv2.INTER_LINEAR)

            # Convert to (C,H,W)
            slice_25d = slice_25d.transpose(2, 0, 1)

            # Normalize
            max_val = np.max(slice_25d)
            if max_val > 0:
                slice_25d = slice_25d / max_val

            slice_25d = torch.tensor(slice_25d, dtype=torch.float32)

        except Exception as e:
            # If MRI file is corrupted, return blank image instead
            slice_25d = torch.zeros((3, 224, 224), dtype=torch.float32)

        label = torch.tensor(self.label, dtype=torch.long)

        return slice_25d, label