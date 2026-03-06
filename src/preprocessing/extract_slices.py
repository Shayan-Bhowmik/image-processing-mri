import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

# Folder containing MRI files
folder_path = r"C:\oasis\OASIS_Clean_Data"

files = os.listdir(folder_path)
file_path = os.path.join(folder_path, files[0])

print("Loading:", file_path)

# Load MRI
mri = nib.load(file_path)
volume = mri.get_fdata()

print("MRI shape:", volume.shape)

# remove channel dimension if present
if volume.ndim == 4:
    volume = volume[:, :, :, 0]

# get number of slices
num_slices = volume.shape[2]

# choose central slices
start = num_slices // 2 - 10
end = num_slices // 2 + 10

slices_2_5d = []

for i in range(start, end):

    prev_slice = volume[:, :, i-1]
    curr_slice = volume[:, :, i]
    next_slice = volume[:, :, i+1]

    # stack slices to form 3-channel image
    slice_2_5d = np.stack([prev_slice, curr_slice, next_slice], axis=-1)

    slices_2_5d.append(slice_2_5d)

print("Total 2.5D slices created:", len(slices_2_5d))

# show one example
example = slices_2_5d[10]

plt.imshow(example[:, :, 1], cmap="gray")
plt.title("Central slice from 2.5D stack")
plt.axis("off")
plt.show()