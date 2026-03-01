import torch
from torch.utils.data import Dataset
import json
import os

from src.preprocessing.load_nifti import load_nifti
from src.preprocessing.normalize import zscore_normalize
from src.preprocessing.slice_extraction import extract_valid_slices
from src.preprocessing.resize import resize_sample


def load_split(split_path, split_name):
    with open(split_path, "r") as f:
        split_dict = json.load(f)
    return split_dict[split_name]


class MRIDataset(Dataset):
    def __init__(self, split_entries, image_size=(224, 224)):
        self.image_size = image_size
        self.index_map = []  # (patient_id, slice_index, label)
        self.volume_store = {}  # patient_id -> normalized 3D volume

        self.brats_root = "data/raw/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
        self.oasis_root = "data/raw/oasis/OASIS_Clean_Data/OASIS_Clean_Data"

        print("Loading and normalizing volumes...")

        for entry in split_entries:
            patient_id = entry["id"]
            label = entry["label"]

            if label == 1:
                patient_path = os.path.join(self.brats_root, patient_id)

                flair_path = None
                for file in os.listdir(patient_path):
                    if "flair" in file.lower() and file.endswith(".nii"):
                        flair_path = os.path.join(patient_path, file)
                        break

                if flair_path is None:
                    continue
            else:
                flair_path = os.path.join(self.oasis_root, patient_id)

            # Load volume once
            volume = load_nifti(flair_path)

            if len(volume.shape) == 4:
                volume = volume[..., 0]

            volume = zscore_normalize(volume)

            # Extract valid slice indices
            slices = extract_valid_slices(volume)
            valid_indices = list(range(len(slices)))

            # Store normalized volume
            self.volume_store[patient_id] = volume

            for slice_idx in valid_indices:
                self.index_map.append((patient_id, slice_idx, label))

        print(f"Total slices indexed: {len(self.index_map)}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        patient_id, slice_idx, label = self.index_map[index]

        volume = self.volume_store[patient_id]

        # Handle boundaries safely
        prev_idx = max(slice_idx - 1, 0)
        next_idx = min(slice_idx + 1, volume.shape[2] - 1)

        slice_prev = volume[:, :, prev_idx]
        slice_curr = volume[:, :, slice_idx]
        slice_next = volume[:, :, next_idx]

        sample = torch.stack([
            torch.from_numpy(slice_prev),
            torch.from_numpy(slice_curr),
            torch.from_numpy(slice_next)
        ], dim=0)

        sample = resize_sample(sample, size=self.image_size)

        return sample, label