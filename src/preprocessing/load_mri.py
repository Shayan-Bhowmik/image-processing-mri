import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# absolute MRI path
file_path = r"C:\brats\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_flair.nii"

# load MRI volume
mri = nib.load(file_path)

# convert to numpy array
mri_data = mri.get_fdata()

print("MRI shape:", mri_data.shape)

# total slices
num_slices = mri_data.shape[2]
print("Total slices:", num_slices)

# show middle slice (verification)
mid = num_slices // 2
slice_img = mri_data[:, :, mid]

plt.imshow(slice_img, cmap="gray")
plt.title(f"Slice {mid}")
plt.axis("off")
plt.show()