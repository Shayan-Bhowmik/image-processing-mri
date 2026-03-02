import numpy as np

from src.dataset.dataset_builder import build_dataset_from_volumes


BRATS_PATH = r"C:\datasets\brats\brats20-dataset-training-validation\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_flair.nii"

OASIS_PATH = r"C:\datasets\oasis\OASIS_Clean_Data\OASIS_Clean_Data\OAS1_0028_MR1_mpr_n4_anon_111_t88_masked_gfc.nii"


def test_dataset_builder():
    print("\n===== Testing Dataset Builder =====")

    sample_list = [
        (BRATS_PATH, 1, "BraTS20_Training_001", "BraTS"),
        (OASIS_PATH, 0, "OAS1_0028_MR1", "OASIS"),
    ]

    dataset = build_dataset_from_volumes(sample_list)

    print("Total slice records:", len(dataset))

    if len(dataset) == 0:
        print("ERROR: Dataset is empty.")
        return

    sample = dataset[0]

    print("\nSample Record Keys:", sample.keys())
    print("Slice shape:", sample["slice"].shape)
    print("Label:", sample["label"])
    print("Patient ID:", sample["patient_id"])
    print("Slice Index:", sample["slice_index"])
    print("Dataset Name:", sample["dataset"])

    # Additional integrity checks
    print("\nData Type of Slice:", sample["slice"].dtype)
    print("Min value:", np.min(sample["slice"]))
    print("Max value:", np.max(sample["slice"]))

    print("\nDataset Construction Test Completed Successfully.")


if __name__ == "__main__":
    test_dataset_builder()