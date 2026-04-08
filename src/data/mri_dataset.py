import gc
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing.load_nifti import load_nifti
from src.preprocessing.normalize import zscore_normalize
from src.preprocessing.resample_3d import resample_volume_3d
from src.preprocessing.resize import resize_sample
from src.preprocessing.slice_extraction import extract_valid_slices


def load_split(split_path, split_name):
    with open(split_path, "r") as f:
        split_dict = json.load(f)
    return split_dict[split_name]


class MRIDataset(Dataset):
    """
    Flexible multi-modal MRI dataset with lazy loading.

    Supports:
    - BRATS 2020 + BRATS 2021 for tumor class
    - OASIS for healthy class
    - Variable number of modalities across datasets

    Uses lazy loading to avoid memory issues with large datasets.
    """

    def __init__(
        self,
        split_entries,
        image_size=(224, 224),
        use_2_5d=True,
        canonical_shape=(192, 192, 160),
        fixed_slice_count=96,
        use_multimodal=True,
        target_modalities=None,
        exclude_brats2021=False,
        use_preprocessed=True,
        preprocessed_root="data/preprocessed",
    ):
        self.image_size = image_size
        self.use_2_5d = use_2_5d
        self.canonical_shape = canonical_shape
        self.fixed_slice_count = fixed_slice_count
        self.use_multimodal = use_multimodal
        self.target_modalities = target_modalities
        self.use_preprocessed = use_preprocessed
        self.preprocessed_root = preprocessed_root

        self.index_map = []
        self.patient_modality_paths = {}
        self.patient_valid_slices = {}
        self.modality_names = []

        self.volume_cache = {}

        self.brats_roots = [
            "data/raw/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData",
        ]
        if not exclude_brats2021:
            self.brats_roots.append("data/raw/brats2021_extracted")
        
        self.oasis_root = "data/raw/oasis/OASIS_Clean_Data/OASIS_Clean_Data"
        self.excluded_patient_ids = {"BraTS20_Training_350"}

        self.all_possible_modalities = ["t1c", "t1", "t2", "flair", "dti", "pwi", "asl"]

        print("Loading dataset metadata (lazy loading mode)...")
        datasets_used = "BRATS 2020 + OASIS" if exclude_brats2021 else "BRATS 2020 + BRATS 2021 + OASIS"
        print(f"Using datasets: {datasets_used}")
        print("Auto-detecting available modalities...\n")

        modalities_found = set()

        for entry in split_entries:
            patient_id = entry["id"]
            label = entry["label"]

            if patient_id in self.excluded_patient_ids:
                continue

            if label == 1:
                patient_path = self._find_patient_path(patient_id)
                if patient_path is None:
                    continue
            else:
                patient_path = os.path.join(self.oasis_root, patient_id)

            modality_paths = self._find_modality_files(patient_path, label)
            if not modality_paths:
                continue

            modalities_found.update(modality_paths.keys())

            self.patient_modality_paths[patient_id] = {
                "paths": modality_paths,
                "label": label,
            }

            # Skip loading volumes during init; build slice index lazily.
            estimated_valid_slices = self.fixed_slice_count if self.fixed_slice_count > 0 else 96
            for valid_idx in range(estimated_valid_slices):
                self.index_map.append((patient_id, valid_idx, label))

        self.modality_names = sorted(modalities_found)
        if self.target_modalities:
            self.modality_names = [m for m in self.target_modalities if m in self.modality_names]

        print(f"Modalities found in dataset: {self.modality_names}")
        print(f"Total modalities: {len(self.modality_names)}")
        print(f"Total slices indexed: {len(self.index_map)}")
        if self.use_preprocessed:
            print(f"Preprocessed cache enabled: {self.preprocessed_root}")
        print("Using lazy loading - volumes loaded on-demand during training.\n")

    def _find_patient_path(self, patient_id):
        for brats_root in self.brats_roots:
            patient_path = os.path.join(brats_root, patient_id)
            if os.path.exists(patient_path):
                return patient_path
        return None

    def _find_modality_files(self, patient_path, label):
        modality_paths = {}

        if label == 1:
            if not os.path.exists(patient_path):
                return modality_paths

            try:
                files = os.listdir(patient_path)
            except OSError:
                return modality_paths

            for file in files:
                file_lower = file.lower()
                if not file_lower.endswith((".nii", ".nii.gz")):
                    continue

                for modality in self.all_possible_modalities:
                    if modality in file_lower and modality not in modality_paths:
                        modality_paths[modality] = os.path.join(patient_path, file)
                        break
        else:
            if os.path.exists(patient_path) and os.path.isfile(patient_path):
                # OASIS files in this workspace are T1-weighted MPR volumes.
                modality_paths["t1"] = patient_path

        return modality_paths

    def _load_patient_volumes(self, patient_id):
        if patient_id in self.volume_cache:
            return self.volume_cache[patient_id]

        modality_volumes = {}
        patient_info = self.patient_modality_paths[patient_id]
        modality_paths = patient_info["paths"]

        preprocessed_path = os.path.join(self.preprocessed_root, f"{patient_id}.npz")
        if self.use_preprocessed and os.path.exists(preprocessed_path):
            try:
                with np.load(preprocessed_path) as data:
                    for modality in data.files:
                        volume = data[modality].astype(np.float32, copy=False)
                        if len(volume.shape) == 4:
                            volume = volume[..., 0]
                        modality_volumes[modality] = volume
            except Exception as e:
                print(f"Warning: Failed to load preprocessed NPZ for {patient_id}: {e}. Falling back to raw NIfTI.")
                modality_volumes = {}

        if modality_volumes:
            self.volume_cache[patient_id] = modality_volumes
            if len(self.volume_cache) > 1:
                oldest_patient = next(iter(self.volume_cache))
                del self.volume_cache[oldest_patient]
                gc.collect()
            return modality_volumes

        for modality, path in modality_paths.items():
            try:
                volume = load_nifti(path)
                if len(volume.shape) == 4:
                    volume = volume[..., 0]
                volume = resample_volume_3d(volume, target_shape=self.canonical_shape)
                volume = zscore_normalize(volume)
                modality_volumes[modality] = volume
            except Exception as e:
                print(f"Warning: Failed to load {modality} for {patient_id}: {e}. Using zero-volume fallback.")
                modality_volumes[modality] = np.zeros(self.canonical_shape, dtype=np.float32)

        self.volume_cache[patient_id] = modality_volumes

        if len(self.volume_cache) > 1:
            oldest_patient = next(iter(self.volume_cache))
            del self.volume_cache[oldest_patient]
            gc.collect()

        return modality_volumes

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        patient_id, valid_idx, label = self.index_map[index]
        modality_volumes = self._load_patient_volumes(patient_id)

        # Compute valid slices lazily if not already done
        if patient_id not in self.patient_valid_slices:
            reference_modality = next(iter(modality_volumes.keys()))
            volume = modality_volumes[reference_modality]
            valid_slices_ref = extract_valid_slices(volume, fixed_count=self.fixed_slice_count)
            self.patient_valid_slices[patient_id] = valid_slices_ref if valid_slices_ref else [np.zeros(self.canonical_shape[:2], dtype=np.float32)]
        
        valid_slices_ref = self.patient_valid_slices[patient_id]

        modality_slices = []
        fallback_shape = self.canonical_shape[:2]
        
        # Clamp valid_idx to actual range of valid slices
        actual_valid_idx = min(valid_idx, len(valid_slices_ref) - 1) if valid_slices_ref else 0

        for modality in self.modality_names:
            if modality in modality_volumes:
                volume = modality_volumes[modality]
                valid_slices = extract_valid_slices(volume, fixed_count=self.fixed_slice_count)
                if len(valid_slices) != len(valid_slices_ref):
                    valid_slices = valid_slices_ref
                slice_curr = valid_slices[actual_valid_idx]
                modality_slices.append(slice_curr)
            else:
                if modality_slices:
                    modality_slices.append(np.zeros_like(modality_slices[0]))
                else:
                    modality_slices.append(np.zeros(fallback_shape, dtype=np.float32))

        sample = torch.stack([torch.from_numpy(s) for s in modality_slices], dim=0)
        sample = resize_sample(sample, size=self.image_size)

        return sample, label, patient_id
