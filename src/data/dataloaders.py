from torch.utils.data import DataLoader
from src.data.mri_dataset import MRIDataset, load_split


def create_dataloaders(
    split_path,
    batch_size=8,
    num_workers=0,
    use_2_5d=True,
    canonical_shape=(192, 192, 160),
    fixed_slice_count=96,
):
    train_paths = load_split(split_path, "train")
    val_paths = load_split(split_path, "val")
    test_paths = load_split(split_path, "test")

    train_dataset = MRIDataset(
        train_paths,
        use_2_5d=use_2_5d,
        canonical_shape=canonical_shape,
        fixed_slice_count=fixed_slice_count,
    )
    val_dataset = MRIDataset(
        val_paths,
        use_2_5d=use_2_5d,
        canonical_shape=canonical_shape,
        fixed_slice_count=fixed_slice_count,
    )
    test_dataset = MRIDataset(
        test_paths,
        use_2_5d=use_2_5d,
        canonical_shape=canonical_shape,
        fixed_slice_count=fixed_slice_count,
    )

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