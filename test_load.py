from src.preprocessing.load_nifti import load_nifti

# Replace this with the FULL path to a BRATS FLAIR file
flair_path = r"C:\Shayan\coding\pbl\image-processing-mri\data\raw\brats\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_flair.nii"

volume = load_nifti(flair_path)

print("Shape:", volume.shape)
print("Dtype:", volume.dtype)