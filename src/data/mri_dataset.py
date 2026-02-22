import torch
from torch.utils.data import Dataset
import json
import os

from src.preprocessing.load_nifti import load_nifti
from src.preprocessing.normalize import zscore_normalize
from src.preprocessing.slice_extraction import extract_valid_slices
from src.preprocessing.stacking import create_25d_samples
from src.preprocessing.resize import resize_sample

from src.label_utils import get_label_from_patient_folder


def load_split(split_path, split_name):
    with open(split_path, "r") as f:
        split_dict = json.load(f)

    return split_dict[split_name]


class MRIDataset(Dataset):
    def __init__(self, patient_paths, image_size=(224, 224)):
        self.patient_paths = patient_paths
        self.image_size = image_size
        self.index_map = []  # (patient_path, slice_index, label)

        for patient_path in patient_paths:
            label = get_label_from_patient_folder(patient_path)

            # Locate FLAIR file
            flair_path = None
            for file in os.listdir(patient_path):
                if "flair" in file.lower() and file.endswith(".nii"):
                    flair_path = os.path.join(patient_path, file)
                    break

            if flair_path is None:
                continue

            # Load once to count valid slices
            volume = load_nifti(flair_path)
            volume = zscore_normalize(volume)
            slices = extract_valid_slices(volume)

            for slice_idx in range(len(slices)):
                self.index_map.append((patient_path, slice_idx, label))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        patient_path, slice_idx, label = self.index_map[index]

        # Locate FLAIR file again
        flair_path = None
        for file in os.listdir(patient_path):
            if "flair" in file.lower() and file.endswith(".nii"):
                flair_path = os.path.join(patient_path, file)
                break

        if flair_path is None:
            raise ValueError(f"FLAIR file not found in {patient_path}")

        # Full preprocessing
        volume = load_nifti(flair_path)
        volume = zscore_normalize(volume)
        slices = extract_valid_slices(volume)
        samples = create_25d_samples(slices)

        sample = samples[slice_idx]
        sample = resize_sample(sample, size=self.image_size)

        return sample, label