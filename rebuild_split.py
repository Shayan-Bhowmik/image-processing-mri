import os
import json
import random

random.seed(42)

brats_root = "data/raw/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
oasis_root = "data/raw/oasis/OASIS_Clean_Data/OASIS_Clean_Data"

# BRATS patients (folders)
brats_patients = [
    {"id": p, "label": 1}
    for p in os.listdir(brats_root)
    if os.path.isdir(os.path.join(brats_root, p))
]

# OASIS patients (files)
oasis_patients = [
    {"id": f, "label": 0}
    for f in os.listdir(oasis_root)
    if f.endswith(".nii")
]

all_patients = brats_patients + oasis_patients
random.shuffle(all_patients)

total = len(all_patients)
train_end = int(0.7 * total)
val_end = int(0.85 * total)

split = {
    "train": all_patients[:train_end],
    "val": all_patients[train_end:val_end],
    "test": all_patients[val_end:]
}

with open("data/splits/patient_split.json", "w") as f:
    json.dump(split, f, indent=4)

print("Total patients:", total)
print("Train:", len(split["train"]))
print("Val:", len(split["val"]))
print("Test:", len(split["test"]))