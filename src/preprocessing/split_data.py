import os
import random

# DATA PATHS
oasis_path = r"C:\oasis\OASIS_Clean_Data"
brats_root = r"C:\brats\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"


# Collect OASIS MRI files (normal)
oasis_files = [
    os.path.join(oasis_path, f)
    for f in os.listdir(oasis_path)
    if f.endswith(".nii")
]


# Collect BraTS T1CE files (abnormal)
brats_files = []

for patient in os.listdir(brats_root):

    patient_path = os.path.join(brats_root, patient)

    if os.path.isdir(patient_path):

        for f in os.listdir(patient_path):

            if "t1ce" in f and f.endswith(".nii"):

                brats_files.append(os.path.join(patient_path, f))


print("OASIS scans:", len(oasis_files))
print("BraTS scans:", len(brats_files))


random.shuffle(oasis_files)
random.shuffle(brats_files)


def split(files):

    train = files[:int(0.7 * len(files))]
    val = files[int(0.7 * len(files)):int(0.85 * len(files))]
    test = files[int(0.85 * len(files)):]

    return train, val, test


oasis_train, oasis_val, oasis_test = split(oasis_files)
brats_train, brats_val, brats_test = split(brats_files)


print("Dataset split complete")