from src.label_utils import (
    get_label_from_patient_folder,
    get_normal_label_from_oasis
)




brats_path = "data/raw/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001"

brats_label = get_label_from_patient_folder(brats_path)
print("BRATS Label:", brats_label)





oasis_file = "data/raw/oasis/OASIS_Clean_Data/OASIS_Clean_Data/OAS1_0001_MR1_mpr_n4_anon_111_t88_masked_gfc.nii"

oasis_label = get_normal_label_from_oasis(oasis_file)
print("OASIS Label:", oasis_label)