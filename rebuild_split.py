import os
import json
import random
import re
from collections import defaultdict

random.seed(42)

brats_roots = [
    "data/raw/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData",
    "data/raw/brats2021_extracted",
]
oasis_root = "data/raw/oasis/OASIS_Clean_Data/OASIS_Clean_Data"


def split_list(items, train_ratio=0.7, val_ratio=0.15):
    total = len(items)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)
    return items[:train_end], items[train_end:val_end], items[val_end:]


def collect_brats_patients(roots):
    patients = []
    for root in roots:
        if not os.path.exists(root):
            continue
        for p in os.listdir(root):
            if os.path.isdir(os.path.join(root, p)):
                patients.append({"id": p, "label": 1})

    # Keep unique ids only (safe if same id appears in multiple roots)
    unique = {}
    for entry in patients:
        unique[entry["id"]] = entry
    return sorted(unique.values(), key=lambda x: x["id"])


brats_patients = collect_brats_patients(brats_roots)


oasis_files = [
    f
    for f in os.listdir(oasis_root)
    if f.lower().endswith((".nii", ".nii.gz"))
]

# Group OASIS scans by subject so MR1/MR2 never cross splits.
subject_pattern = re.compile(r"^(OAS\d?_\d{4})_")
oasis_by_subject = defaultdict(list)

for file_name in oasis_files:
    match = subject_pattern.match(file_name)
    subject_id = match.group(1) if match else file_name
    oasis_by_subject[subject_id].append(file_name)

for subject_id in oasis_by_subject:
    oasis_by_subject[subject_id].sort()

oasis_subjects = sorted(oasis_by_subject.keys())

random.shuffle(brats_patients)
random.shuffle(oasis_subjects)

brats_train, brats_val, brats_test = split_list(brats_patients)
oasis_train_subjects, oasis_val_subjects, oasis_test_subjects = split_list(oasis_subjects)


def expand_oasis(subject_ids):
    entries = []
    for subject_id in subject_ids:
        for file_name in oasis_by_subject[subject_id]:
            entries.append({"id": file_name, "label": 0})
    return entries


train_entries = brats_train + expand_oasis(oasis_train_subjects)
val_entries = brats_val + expand_oasis(oasis_val_subjects)
test_entries = brats_test + expand_oasis(oasis_test_subjects)

random.shuffle(train_entries)
random.shuffle(val_entries)
random.shuffle(test_entries)

total = len(train_entries) + len(val_entries) + len(test_entries)

split = {
    "train": train_entries,
    "val": val_entries,
    "test": test_entries,
}

with open("data/splits/patient_split.json", "w") as f:
    json.dump(split, f, indent=4)

print("Total patients:", total)
print("Train:", len(split["train"]))
print("Val:", len(split["val"]))
print("Test:", len(split["test"]))
print("OASIS subjects:", len(oasis_subjects))