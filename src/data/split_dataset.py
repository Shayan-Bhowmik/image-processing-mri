import os
import random
import json


def get_patient_folders(root_dir):
    patient_paths = []

    for item in os.listdir(root_dir):
        full_path = os.path.join(root_dir, item)

        if os.path.isdir(full_path):
            patient_paths.append(full_path)

    return sorted(patient_paths)


def split_patients(patient_paths, train_ratio=0.7, val_ratio=0.15, seed=42):
    random.seed(seed)
    shuffled = patient_paths.copy()
    random.shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]

    return train, val, test


def save_split(train, val, test, save_path):
    split_dict = {
        "train": train,
        "val": val,
        "test": test
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(split_dict, f, indent=4)


if __name__ == "__main__":
    root_dir = "data/raw/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"  # adjust if needed

    patient_paths = get_patient_folders(root_dir)
    train, val, test = split_patients(patient_paths)

    save_split(
        train,
        val,
        test,
        save_path="data/splits/patient_split.json"
    )

    print(f"Total patients: {len(patient_paths)}")
    print(f"Train: {len(train)}")
    print(f"Val: {len(val)}")
    print(f"Test: {len(test)}")