import pickle
import gzip
from pathlib import Path

from src.dataset.dataset_builder import build_dataset_from_volumes


def get_brats_volumes(brats_root):
    volumes = []

    for patient in Path(brats_root).iterdir():
        if patient.is_dir():
            for file in patient.glob("*flair*.nii*"):
                volumes.append((str(file), 1, patient.name, "brats"))

    return volumes


def get_oasis_volumes(oasis_root):
    volumes = []

    for file in Path(oasis_root).rglob("*.nii*"):

        if "seg" in file.name.lower():
            continue

        volumes.append((str(file), 0, file.stem, "oasis"))

    return volumes


def main():

    brats_root = r"C:\datasets\brats\brats20-dataset-training-validation\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
    oasis_root = r"C:\datasets\oasis\OASIS_Clean_Data\OASIS_Clean_Data"

    print("Scanning datasets...")

    brats_volumes = get_brats_volumes(brats_root)
    oasis_volumes = get_oasis_volumes(oasis_root)

    print(f"BraTS volumes: {len(brats_volumes)}")
    print(f"OASIS volumes: {len(oasis_volumes)}")

    if len(brats_volumes) == 0 or len(oasis_volumes) == 0:
        raise ValueError("Dataset scanning failed. Check dataset paths.")

    all_volumes = brats_volumes + oasis_volumes

    print("Building dataset records... (this will take time)")

    dataset_records = build_dataset_from_volumes(all_volumes)

    print(f"Total slice records: {len(dataset_records)}")

    output_path = Path("data/dataset_records.pkl.gz")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(output_path, "wb") as f:
        pickle.dump(dataset_records, f)

    print(f"Dataset saved to: {output_path}")


if __name__ == "__main__":
    main()