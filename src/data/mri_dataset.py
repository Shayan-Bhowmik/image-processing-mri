import torch
from torch.utils.data import Dataset
import json
import os

from src.preprocessing.load_nifti import load_nifti
from src.preprocessing.normalize import zscore_normalize
from src.preprocessing.slice_extraction import extract_valid_slices
from src.preprocessing.stacking import create_25d_samples
from src.preprocessing.resize import resize_sample


def load_split(split_path, split_name):
    with open(split_path, "r") as f:
        split_dict = json.load(f)

    return split_dict[split_name]


class MRIDataset(Dataset):
    def __init__(self, split_entries, image_size=(224, 224)):
        """
        split_entries: list of dicts from patient_split.json
        Each dict: {"id": ..., "label": 0/1}
        """

        self.split_entries = split_entries
        self.image_size = image_size
        self.index_map = []  # (patient_id, slice_index, label)
        self.volume_cache = {}

        self.brats_root = "data/raw/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
        self.oasis_root = "data/raw/oasis/OASIS_Clean_Data/OASIS_Clean_Data"

        # Build index map
        for entry in split_entries:
            patient_id = entry["id"]
            label = entry["label"]

            # Determine file path
            if label == 1:
                # BRATS (patient folder)
                patient_path = os.path.join(self.brats_root, patient_id)

                flair_path = None
                for file in os.listdir(patient_path):
                    if "flair" in file.lower() and file.endswith(".nii"):
                        flair_path = os.path.join(patient_path, file)
                        break

                if flair_path is None:
                    continue
            else:
                # OASIS (direct .nii file)
                flair_path = os.path.join(self.oasis_root, patient_id)

            # Load volume
            volume = load_nifti(flair_path)

            # Fix 4D volumes (common in OASIS)
            if len(volume.shape) == 4:
                volume = volume[..., 0]

            volume = zscore_normalize(volume)
            slices = extract_valid_slices(volume)

            for slice_idx in range(len(slices)):
                self.index_map.append((patient_id, slice_idx, label))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        patient_id, slice_idx, label = self.index_map[index]

        # Reconstruct file path
        if label == 1:
            patient_path = os.path.join(self.brats_root, patient_id)

            flair_path = None
            for file in os.listdir(patient_path):
                if "flair" in file.lower() and file.endswith(".nii"):
                    flair_path = os.path.join(patient_path, file)
                    break
        else:
            flair_path = os.path.join(self.oasis_root, patient_id)

        if flair_path is None:
            raise ValueError(f"FLAIR file not found for {patient_id}")

        # Use cache if available
        if patient_id in self.volume_cache:
            samples = self.volume_cache[patient_id]
        else:
            volume = load_nifti(flair_path)

            # Fix 4D volumes
            if len(volume.shape) == 4:
                volume = volume[..., 0]

            volume = zscore_normalize(volume)
            slices = extract_valid_slices(volume)
            samples = create_25d_samples(slices)

            self.volume_cache[patient_id] = samples

        sample = samples[slice_idx]
        sample = resize_sample(sample, size=self.image_size)

        return sample, label