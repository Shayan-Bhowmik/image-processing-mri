from src.label_utils import get_label_from_patient_folder

patient_path = "data/raw/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001"

label = get_label_from_patient_folder(patient_path)

print("Label:", label)