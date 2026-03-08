import glob
import os
import torch
from torch.utils.data import DataLoader, ConcatDataset

from src.models.resnet_model import build_model
from src.models.mri_dataset import MRIDataset
from src.training.train_utils import get_training_components
from src.training.train_model import train_one_epoch
from src.training.validate import validate
from src.utils.save_model import save_model


def main():

    # =====================
    # DATA PATHS
    # =====================

    oasis_path = r"C:\oasis\OASIS_Clean_Data"

    brats_path = r"C:\brats\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"

    # =====================
    # COLLECT FILES
    # =====================

    normal_files = glob.glob(os.path.join(oasis_path, "*.nii"))

    tumor_files = glob.glob(os.path.join(brats_path, "*", "*flair.nii"))

    print("Normal MRI:", len(normal_files))
    print("Tumor MRI:", len(tumor_files))

    # =====================
    # DATASETS
    # =====================

    normal_dataset = MRIDataset(normal_files, 0)
    tumor_dataset = MRIDataset(tumor_files, 1)

    dataset = ConcatDataset([normal_dataset, tumor_dataset])

    # =====================
    # DATALOADER
    # =====================

    train_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True
    )

    # =====================
    # MODEL
    # =====================

    model = build_model()

    model, criterion, optimizer, device = get_training_components(model)

    print("Training on:", device)

    epochs = 5

    # =====================
    # TRAINING LOOP
    # =====================

    for epoch in range(epochs):

        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_acc:.4f}")

    # =====================
    # SAVE MODEL
    # =====================

    os.makedirs("models", exist_ok=True)

    save_model(model, "models/mri_resnet50.pth")

    print("\nModel saved to models/mri_resnet50.pth")


if __name__ == "__main__":
    main()