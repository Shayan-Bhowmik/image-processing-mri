from src.preprocessing.load_nifti import load_nifti
from src.preprocessing.normalize import zscore_normalize
from src.preprocessing.slice_extraction import extract_valid_slices
from src.preprocessing.stacking import create_25d_samples
from src.preprocessing.resize import resize_sample

flair_path = r"C:\Shayan\coding\pbl\image-processing-mri\data\raw\brats\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_flair.nii"

volume = load_nifti(flair_path)
volume = zscore_normalize(volume)

slices = extract_valid_slices(volume, threshold=0.01)
samples_25d = create_25d_samples(slices)

resized_sample = resize_sample(samples_25d[0])

print("Original 2.5D Shape:", samples_25d[0].shape)
print("Resized Shape:", resized_sample.shape)
print("Tensor Type:", type(resized_sample))