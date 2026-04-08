"""
Pre-process MRI volumes to efficient NPZ format for fast training.
Eliminates runtime resampling/normalization bottleneck.
"""

import json
import os
import numpy as np
from tqdm import tqdm
from src.preprocessing.load_nifti import load_nifti
from src.preprocessing.resample_3d import resample_volume_3d
from src.preprocessing.normalize import zscore_normalize

def preprocess_dataset(output_dir="data/preprocessed"):
    """Convert all raw NIfTI volumes to NPZ format."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    brats_root = "data/raw/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    oasis_root = "data/raw/oasis/OASIS_Clean_Data/OASIS_Clean_Data"
    
    all_modalities = ["t1c", "t1", "t2", "flair", "dti", "pwi", "asl"]
    canonical_shape = (192, 192, 160)
    excluded = {"BraTS20_Training_350"}
    
    # Load split to know which patients to process
    with open("data/splits/patient_split.json") as f:
        split_data = json.load(f)
    
    all_patients = {}
    for split_name in ["train", "val", "test"]:
        for entry in split_data[split_name]:
            patient_id = entry["id"]
            label = entry["label"]
            all_patients[patient_id] = label
    
    processed_count = 0
    processed_brats2020 = 0
    processed_oasis = 0
    skipped_existing = 0
    
    for patient_id, label in tqdm(all_patients.items(), desc="Preprocessing volumes"):
        if patient_id in excluded:
            continue
        
        out_path = os.path.join(output_dir, f"{patient_id}.npz")
        if os.path.exists(out_path):
            skipped_existing += 1
            continue
        
        modality_volumes = {}
        
        if label == 1:
            # BRATS patient (BraTS2020 only for preprocessing)
            patient_path = os.path.join(brats_root, patient_id)
            
            if not os.path.exists(patient_path):
                continue
            
            try:
                files = os.listdir(patient_path)
            except OSError:
                continue
            
            for file in files:
                if not file.lower().endswith((".nii", ".nii.gz")):
                    continue
                
                file_lower = file.lower()
                for modality in all_modalities:
                    if modality in file_lower and modality not in modality_volumes:
                        filepath = os.path.join(patient_path, file)
                        try:
                            vol = load_nifti(filepath)
                            if len(vol.shape) == 4:
                                vol = vol[..., 0]
                            vol = resample_volume_3d(vol, target_shape=canonical_shape)
                            vol = zscore_normalize(vol)
                            modality_volumes[modality] = vol.astype(np.float32)
                        except Exception as e:
                            print(f"Warning: Failed to load {modality} for {patient_id}: {e}")
                            modality_volumes[modality] = np.zeros(canonical_shape, dtype=np.float32)
        else:
            # OASIS patient
            patient_path = os.path.join(oasis_root, patient_id)
            if not os.path.exists(patient_path):
                continue

            if os.path.isfile(patient_path):
                # OASIS split entries are file paths, not patient directories.
                try:
                    vol = load_nifti(patient_path)
                    if len(vol.shape) == 4:
                        vol = vol[..., 0]
                    vol = resample_volume_3d(vol, target_shape=canonical_shape)
                    vol = zscore_normalize(vol)
                    modality_volumes["t1"] = vol.astype(np.float32)
                except Exception as e:
                    print(f"Warning: Failed to load OASIS volume for {patient_id}: {e}")
                    modality_volumes["t1"] = np.zeros(canonical_shape, dtype=np.float32)
            else:
                try:
                    files = os.listdir(patient_path)
                except OSError:
                    continue

                for file in files:
                    if not file.lower().endswith((".nii", ".nii.gz")):
                        continue

                    file_lower = file.lower()
                    for modality in all_modalities:
                        if modality in file_lower and modality not in modality_volumes:
                            filepath = os.path.join(patient_path, file)
                            try:
                                vol = load_nifti(filepath)
                                if len(vol.shape) == 4:
                                    vol = vol[..., 0]
                                vol = resample_volume_3d(vol, target_shape=canonical_shape)
                                vol = zscore_normalize(vol)
                                modality_volumes[modality] = vol.astype(np.float32)
                            except Exception as e:
                                print(f"Warning: Failed to load {modality} for {patient_id}: {e}")
                                modality_volumes[modality] = np.zeros(canonical_shape, dtype=np.float32)
        
        if modality_volumes:
            np.savez_compressed(out_path, **modality_volumes)
            processed_count += 1
            if label == 1:
                processed_brats2020 += 1
            else:
                processed_oasis += 1
    
    print(f"\nPreprocessing complete! Saved {processed_count} patients to {output_dir}")
    print(f"Saved BraTS2020: {processed_brats2020}")
    print(f"Saved OASIS: {processed_oasis}")
    print(f"Skipped existing NPZ: {skipped_existing}")

if __name__ == "__main__":
    preprocess_dataset()
