import os

brats_root = "data/raw/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
oasis_root = "data/raw/oasis/OASIS_Clean_Data/OASIS_Clean_Data"

print("BRATS exists:", os.path.exists(brats_root))
print("OASIS exists:", os.path.exists(oasis_root))

print("BRATS patients:", len(os.listdir(brats_root)))
print("OASIS patients:", len(os.listdir(oasis_root)))