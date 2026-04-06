import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.models.cnn_model import CNNModel
from src.preprocessing.load_mri import load_nifti
from src.preprocessing.extract_slices import extract_slices
from src.preprocessing.preprocess import normalize, resize_slice
from src.preprocessing.stack_slices import stack_slices


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BRATS_PATH = r"C:\dataset\brats\BraTS2020_TrainingData"
OASIS_PATH = r"C:\dataset\oasis"


class MRIDataset(Dataset):

    def __init__(self, paths, label):

        self.paths = paths
        self.label = label

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        path = self.paths[idx]

        volume = load_nifti(path)

        slices = extract_slices(volume)

        stacked = stack_slices(slices)

        mid = len(stacked) // 2

        slice_group = stacked[mid-5:mid+5]

        s = slice_group[0]   # pick one slice randomly or first slice

        img = normalize(s[1])
        img = resize_slice(img)

        img = torch.tensor(img).float()
        img = img.unsqueeze(0).repeat(3,1,1)

        return img, self.label

def collect_files(folder, keyword):

    paths = []

    for root, _, files in os.walk(folder):

        for f in files:

            if keyword in f and f.endswith(".nii"):
                paths.append(os.path.join(root,f))

    return paths


def build_dataset():

    tumor_files = collect_files(BRATS_PATH,"flair")
    healthy_files = collect_files(OASIS_PATH,".nii")

    tumor_dataset = MRIDataset(tumor_files,1)
    healthy_dataset = MRIDataset(healthy_files,0)

    dataset = torch.utils.data.ConcatDataset([tumor_dataset, healthy_dataset])

    return dataset


def validate(model,loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for x,y in loader:

            x = x.to(device)
            y = y.to(device)

            out = model(x)

            preds = torch.argmax(out,1)

            correct += (preds==y).sum().item()
            total += y.size(0)

    return correct/total


def train():

    dataset = build_dataset()

    from torch.utils.data import random_split

    split = int(0.8 * len(dataset))

    train_data, val_data = random_split(dataset, [split, len(dataset) - split])

    train_loader = DataLoader(train_data,batch_size=8,shuffle=True)
    val_loader = DataLoader(val_data,batch_size=8)

    model = CNNModel().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    best_acc = 0

    for epoch in range(50):

        model.train()

        correct = 0
        total = 0

        for x,y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            out = model(x)

            loss = criterion(out,y)

            loss.backward()

            optimizer.step()

            preds = torch.argmax(out,1)

            correct += (preds==y).sum().item()
            total += y.size(0)

        train_acc = correct/total

        val_acc = validate(model,val_loader)

        print(f"\nEpoch {epoch+1}/50")
        print("Train Acc:",train_acc)
        print("Val Acc:",val_acc)

        if val_acc > best_acc:

            best_acc = val_acc

            torch.save(model.state_dict(),"models/best_model.pth")


if __name__ == "__main__":

    train()