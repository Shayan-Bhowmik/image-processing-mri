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
    def __init__(self, split_entries, image_size=(224, 224), use_2_5d=True):
        self.image_size = image_size
        self.use_2_5d = use_2_5d
        self.index_map = []
        self.volume_store = {}
        self.valid_slices_store = {}

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




            volume = load_nifti(flair_path)

            if len(volume.shape) == 4:
                volume = volume[..., 0]

            volume = zscore_normalize(volume)
            valid_slices = extract_valid_slices(volume)
            if not valid_slices:
                continue

            self.volume_store[patient_id] = volume
            self.valid_slices_store[patient_id] = valid_slices

            for valid_idx in range(len(valid_slices)):
                self.index_map.append((patient_id, valid_idx, label))

        print(f"Total slices indexed: {len(self.index_map)}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        patient_id, valid_idx, label = self.index_map[index]

        valid_slices = self.valid_slices_store[patient_id]

        prev_idx = max(valid_idx - 1, 0)
        next_idx = min(valid_idx + 1, len(valid_slices) - 1)

        slice_prev = valid_slices[prev_idx]
        slice_curr = valid_slices[valid_idx]
        slice_next = valid_slices[next_idx]




        if self.use_2_5d:
            sample = torch.stack([
                torch.from_numpy(slice_prev),
                torch.from_numpy(slice_curr),
                torch.from_numpy(slice_next)
            ], dim=0)
        else:
            sample = torch.from_numpy(slice_curr).unsqueeze(0)


        sample = resize_sample(sample, size=self.image_size)

        return sample, label, patient_id