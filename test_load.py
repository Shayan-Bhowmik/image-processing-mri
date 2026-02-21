from src.preprocessing.load_nifti import load_nifti
from src.preprocessing.normalize import zscore_normalize
from src.preprocessing.slice_extraction import extract_valid_slices
from src.preprocessing.stacking import create_25d_samples

flair_path = r"C:\Shayan\coding\pbl\image-processing-mri\data\raw\brats\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_flair.nii"

volume = load_nifti(flair_path)
volume = zscore_normalize(volume)

slices = extract_valid_slices(volume, threshold=0.01)
samples_25d = create_25d_samples(slices)

print("Valid 2D Slices:", len(slices))
print("2.5D Samples:", len(samples_25d))

if len(samples_25d) > 0:
    print("Single 2.5D Sample Shape:", samples_25d[0].shape)