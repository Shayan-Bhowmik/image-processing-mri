import numpy as np
from src.preprocessing.volume_utils import load_nifti, zscore_normalize

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

    mask = volume!= 0

    if np.sum(mask) > 0:
        print("\nAfter Normalization (non-zero region):")
        print("Mean:", np.mean(normalized[mask]))
        print("Std:", np.std(normalized[mask]))
    else:
        print("Warning: No non-zero voxels found.")

if __name__ == "__main__":
    test_volume(BRATS_PATH, "BraTS")
    test_volume(OASIS_PATH, "OASIS")