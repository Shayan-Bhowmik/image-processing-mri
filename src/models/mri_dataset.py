import os
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

        # load MRI
        img = nib.load(path)
        volume = img.get_fdata()

        # choose center slice
        slice_index = volume.shape[2] // 2

        prev_slice = volume[:, :, slice_index-1]
        curr_slice = volume[:, :, slice_index]
        next_slice = volume[:, :, slice_index+1]

        # create 2.5D image
        slice_25d = np.stack([prev_slice, curr_slice, next_slice], axis=0)

        # resize
        slice_25d = cv2.resize(slice_25d.transpose(1,2,0),(224,224))

        slice_25d = slice_25d.transpose(2,0,1)

        # normalize
        slice_25d = slice_25d / np.max(slice_25d)

        slice_25d = torch.tensor(slice_25d, dtype=torch.float32)

        label = torch.tensor(self.label, dtype=torch.long)

        return slice_25d, label