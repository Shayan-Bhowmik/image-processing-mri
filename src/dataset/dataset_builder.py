import numpy as np

from ..preprocessing.volume_utils import load_nifti, zscore_normalize
from ..preprocessing.slice_utils import (
    extract_axial_slices,
    is_informative_slice
)


def create_slice_record(slice_2d, label, patient_id, slice_index, dataset_name):
    return {
        "slice": slice_2d,
        "label": label,
        "patient_id": patient_id,
        "slice_index": slice_index,  
        "dataset": dataset_name,
    }


def build_volume_dataset(volume, label, patient_id, dataset_name, threshold=0.05):
    
    if volume.ndim != 3:
        raise ValueError(
            f"Expected 3D volume before dataset construction, got shape {volume.shape}"
        )
    
    slices = extract_axial_slices(volume)
    
    slice_records = []
    
    for original_index, slice_2d in enumerate(slices):
        
        if is_informative_slice(slice_2d, threshold=threshold):
            
            record = create_slice_record(
                slice_2d=slice_2d,
                label=label,
                patient_id=patient_id,
                slice_index=original_index, 
                dataset_name=dataset_name
            )
            
            slice_records.append(record)
    
    return slice_records


def build_dataset_from_volumes(list_of_volumes, threshold=0.05):
    
    master_dataset = []
    
    for volume_path, label, patient_id, dataset_name in list_of_volumes:
        
        volume = load_nifti(volume_path)
        normalized_volume = zscore_normalize(volume)
        
        slice_records = build_volume_dataset(
            volume=normalized_volume,
            label=label,
            patient_id=patient_id,
            dataset_name=dataset_name,
            threshold=threshold
        )
        
        master_dataset.extend(slice_records)
    
    return master_dataset