from src.data.mri_dataset import MRIDataset, load_split
from torch.utils.data import DataLoader

train_entries = load_split("data/splits/patient_split.json", "train")
train_dataset = MRIDataset(train_entries)

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

all_labels = []

for _, labels in train_loader:
    all_labels.extend(labels.tolist())

print("Total slices:", len(all_labels))
print("Class 0 slices:", all_labels.count(0))
print("Class 1 slices:", all_labels.count(1))