"""
Diagnostic script to identify training bottlenecks.
Measures time per batch during training to pinpoint slow operations.
"""

import time
import json
import torch
from torch.utils.data import DataLoader
from src.data.mri_dataset import MRIDataset, load_split

# Load data split
split_path = "data/splits/patient_split.json"
split_entries = load_split(split_path, "train")

print(f"\nLoading {len(split_entries)} train patients...")

# Create dataset
dataset = MRIDataset(
    split_entries=split_entries,
    image_size=(224, 224),
    use_2_5d=True,
    canonical_shape=(192, 192, 160),
    fixed_slice_count=48,
    use_multimodal=True,
)

print(f"\nDataset Info:")
print(f"  Total slices: {len(dataset)}")
print(f"  Modalities: {dataset.modality_names}")
print(f"  Cache size: 2 patients")

# Create dataloader
print(f"\nCreating DataLoader with batch_size=8...")
train_loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
)

print(f"\n{'='*70}")
print("BATCH LOADING DIAGNOSTIC")
print(f"{'='*70}\n")

batch_times = []
total_start = time.time()

for batch_idx, (images, labels, patient_ids) in enumerate(train_loader):
    batch_start = time.time()
    batch_time = batch_start - (total_start if batch_idx == 0 else prev_batch_end)
    batch_times.append(batch_time)
    
    unique_patients = len(set(patient_ids))
    
    print(f"Batch {batch_idx+1}:")
    print(f"  Unique patients in batch: {unique_patients}")
    print(f"  Patient IDs: {list(set(patient_ids))[:3]}{'...' if unique_patients > 3 else ''}")
    print(f"  Batch shape: {images.shape}")
    print(f"  Load time: {batch_time:.2f}s")
    
    prev_batch_end = time.time()
    
    if batch_idx >= 9:  # Test first 10 batches
        break

avg_batch_time = sum(batch_times) / len(batch_times)
print(f"\n{'='*70}")
print(f"Average batch loading time: {avg_batch_time:.2f}s")
print(f"Min: {min(batch_times):.2f}s | Max: {max(batch_times):.2f}s")

if avg_batch_time > 2:
    print("\n⚠️  BOTTLENECK DETECTED: Batch loading is slow!")
    print("   Likely cause: Small cache size (2 patients) causes constant reloading")
    print("   Solution: Increase volume_cache size in src/data/mri_dataset.py")
else:
    print("\n✓ Batch loading speed is acceptable")
