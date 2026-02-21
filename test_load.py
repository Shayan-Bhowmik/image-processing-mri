from src.preprocessing.load_nifti import load_nifti
from src.preprocessing.normalize import zscore_normalize

flair_path = r"C:\Shayan\coding\pbl\image-processing-mri\data\raw\brats\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_flair.nii"

volume = load_nifti(flair_path)

print("Shape:", volume.shape)
print("Dtype:", volume.dtype)

print("\nBefore Normalization:")
print("Mean (all voxels):", volume.mean())
print("Std (all voxels):", volume.std())

# Create mask BEFORE normalization
mask = volume > 0

volume = zscore_normalize(volume)

print("\nAfter Normalization (same non-zero mask):")
print("Mean:", volume[mask].mean())
print("Std:", volume[mask].std())