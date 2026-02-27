from torch.utils.data import DataLoader
from src.data.mri_dataset import MRIDataset, load_split


def create_dataloaders(
    split_path,
    batch_size=8,
    num_workers=0
):
    train_paths = load_split(split_path, "train")
    val_paths = load_split(split_path, "val")
    test_paths = load_split(split_path, "test")

    train_dataset = MRIDataset(train_paths)
    val_dataset = MRIDataset(val_paths)
    test_dataset = MRIDataset(test_paths)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader