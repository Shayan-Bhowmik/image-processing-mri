from src.preprocessing.load_nifti import load_nifti
from src.preprocessing.normalize import zscore_normalize
from src.preprocessing.slice_extraction import extract_valid_slices

flair_path = r"C:\Shayan\coding\pbl\image-processing-mri\data\raw\brats\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_flair.nii"

volume = load_nifti(flair_path)
volume = zscore_normalize(volume)

print("Original Volume Shape:", volume.shape)

slices = extract_valid_slices(volume, threshold=0.01)

print("Total Axial Slices:", volume.shape[2])
print("Valid Slices After Filtering:", len(slices))

if len(slices) > 0:
    print("Single Slice Shape:", slices[0].shape)