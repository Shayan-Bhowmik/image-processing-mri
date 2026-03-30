"""
Analyze the dataset records to understand data distribution
"""

import pickle
import gzip
from pathlib import Path
from collections import Counter

# Load dataset records
dataset_path = Path("data/dataset_records.pkl.gz")

with gzip.open(dataset_path, "rb") as f:
    records = pickle.load(f)

print("=" * 60)
print("DATASET ANALYSIS")
print("=" * 60)

print(f"\nTotal records: {len(records)}")

# Count by label
labels = [r["label"] for r in records]
label_counts = Counter(labels)

print(f"\nLabel distribution:")
print(f"  Normal (0): {label_counts[0]:,} slices ({100*label_counts[0]/len(records):.1f}%)")
print(f"  Tumor (1):  {label_counts[1]:,} slices ({100*label_counts[1]/len(records):.1f}%)")

class_ratio = label_counts[1] / label_counts[0]
print(f"\nClass imbalance ratio (Tumor/Normal): {class_ratio:.2f}x")

# Count by dataset
datasets = [r["dataset"] for r in records]
dataset_counts = Counter(datasets)

print(f"\nDataset distribution:")
for ds, count in dataset_counts.items():
    print(f"  {ds}: {count:,} slices ({100*count/len(records):.1f}%)")

# Count by dataset and label
print(f"\nLabel distribution by dataset:")
for ds in dataset_counts.keys():
    ds_records = [r for r in records if r["dataset"] == ds]
    ds_labels = [r["label"] for r in ds_records]
    ds_label_counts = Counter(ds_labels)
    
    normal = ds_label_counts[0]
    tumor = ds_label_counts[1] if 1 in ds_label_counts else 0
    
    print(f"  {ds}:")
    print(f"    Normal: {normal:,}")
    print(f"    Tumor:  {tumor:,}")

# Count unique patients
patients = set(r["patient_id"] for r in records)
print(f"\nUnique patients: {len(patients)}")

# Patients by label
normal_patients = set(r["patient_id"] for r in records if r["label"] == 0)
tumor_patients = set(r["patient_id"] for r in records if r["label"] == 1)

print(f"  Normal patients: {len(normal_patients)}")
print(f"  Tumor patients:  {len(tumor_patients)}")

print("\n" + "=" * 60)
