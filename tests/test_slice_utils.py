import numpy as np

from src.preprocessing.volume_utils import load_nifti, zscore_normalize
from src.preprocessing.slice_utils import (
    extract_axial_slices,
    filter_empty_slices
)

BRATS_PATH = r"C:\datasets\brats\brats20-dataset-training-validation\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_flair.nii"

OASIS_PATH = r"C:\datasets\oasis\OASIS_Clean_Data\OASIS_Clean_Data\OAS1_0028_MR1_mpr_n4_anon_111_t88_masked_gfc.nii"


def test_volume(path, dataset_name):
    print(f"\n===== Testing {dataset_name} =====")

    volume = load_nifti(path)

    print("After Loading:")
    print("Shape:", volume.shape)
    print("Dtype:", volume.dtype)
    print("Min:", np.min(volume))
    print("Max:", np.max(volume))

    normalized = zscore_normalize(volume)

    mask = volume != 0

    if np.sum(mask) > 0:
        print("\nAfter Normalization (non-zero region):")
        print("Mean:", np.mean(normalized[mask]))
        print("Std:", np.std(normalized[mask]))
    else:
        print("Warning: No non-zero voxels found.")
    slices = extract_axial_slices(normalized)
    filtered_slices = filter_empty_slices(slices)

    print("\nSlice Processing:")
    print("Total slices:", len(slices))
    print("After filtering:", len(filtered_slices))

    if len(slices) > 0:
        retention = (len(filtered_slices) / len(slices)) * 100
        print("Retention %:", round(retention, 2))

    if filtered_slices:
        print("Sample slice shape:", filtered_slices[0].shape)
    else:
        print("No slices retained after filtering.")


if __name__ == "__main__":
    test_volume(BRATS_PATH, "BraTS")
    test_volume(OASIS_PATH, "OASIS")