from src.data.mri_dataset import MRIDataset, load_split

train_entries = load_split("data/splits/patient_split.json", "train")
dataset = MRIDataset(train_entries)

print("Total slices:", len(dataset))

sample, label = dataset[0]
print("Sample shape:", sample.shape)
print("Label:", label)