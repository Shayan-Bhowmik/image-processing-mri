import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple

import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from ..preprocessing.volume_utils import strip_skull


MRI_CHANNEL_MEAN: Tuple[float, float, float] = (0.0, 0.0, 0.0)
MRI_CHANNEL_STD: Tuple[float, float, float] = (1.0, 1.0, 1.0)

DEFAULT_CENTER_CROP_RATIO: float = 0.8


def build_patient_index(dataset: List[Dict]) -> Dict[str, List[Dict]]:
  
    patient_index = {}

    for record in dataset:
        pid = record["patient_id"]

        if pid not in patient_index:
            patient_index[pid] = []

        patient_index[pid].append(record)

    for pid in patient_index:
        patient_index[pid].sort(key=lambda x: x["slice_index"])

    return patient_index


def stack_2_5d(record: Dict, patient_slices: List[Dict], apply_skull_strip: bool = True) -> np.ndarray:
 
    current_index = record["slice_index"]
    slice_map = {r["slice_index"]: r["slice"] for r in patient_slices}

    h, w = record["slice"].shape
    zero_slice = np.zeros((h, w), dtype=np.float32)

    prev_slice = slice_map.get(current_index - 1, zero_slice)
    curr_slice = slice_map.get(current_index)
    next_slice = slice_map.get(current_index + 1, zero_slice)

    if apply_skull_strip:
        prev_slice = strip_skull(prev_slice, margin=20)
        curr_slice = strip_skull(curr_slice, margin=20)
        next_slice = strip_skull(next_slice, margin=20)

    stacked = np.stack([prev_slice, curr_slice, next_slice], axis=-1)

    return stacked


def resize_image(image: np.ndarray, target_size: int) -> np.ndarray:

    resized = cv2.resize(
        image,
        (target_size, target_size),
        interpolation=cv2.INTER_LINEAR
    )

    return resized


def transform_record(
    record: Dict,
    patient_index: Dict[str, List[Dict]],
    target_size: int,
    pre_resize: bool = True,
    apply_skull_strip: bool = True,
) -> np.ndarray:

    pid = record["patient_id"]
    patient_slices = patient_index[pid]

    stacked = stack_2_5d(record, patient_slices, apply_skull_strip=apply_skull_strip)

    if pre_resize:
        return resize_image(stacked, target_size)

    return stacked


def resolve_center_crop_size(
    target_size: int,
    center_crop_size: Optional[int],
) -> int:

    if target_size <= 0:
        raise ValueError(f"target_size must be > 0, got {target_size}")

    if center_crop_size is None:
        center_crop_size = int(round(target_size * DEFAULT_CENTER_CROP_RATIO))

    if center_crop_size <= 0:
        raise ValueError(f"center_crop_size must be > 0, got {center_crop_size}")

    if center_crop_size > target_size:
        raise ValueError(
            f"center_crop_size ({center_crop_size}) cannot exceed target_size ({target_size})"
        )

    return center_crop_size


def validate_normalization(
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> None:

    if len(mean) != 3 or len(std) != 3:
        raise ValueError("mean and std must contain exactly 3 values for 2.5D input")

    if any(s <= 0.0 for s in std):
        raise ValueError(f"All std values must be > 0, got {std}")


def build_train_transform(
    target_size: int = 224,
    center_crop_size: Optional[int] = None,
    mean: Tuple[float, float, float] = MRI_CHANNEL_MEAN,
    std: Tuple[float, float, float] = MRI_CHANNEL_STD
):

    crop_size = resolve_center_crop_size(target_size, center_crop_size)
    validate_normalization(mean, std)

    return T.Compose([
        T.CenterCrop(crop_size),
        T.RandomResizedCrop(
            size=target_size,
            scale=(0.8, 1.0),
            ratio=(0.95, 1.05),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(
            degrees=10,
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        ),
        T.Normalize(mean=mean, std=std),
    ])


def build_eval_transform(
    target_size: int = 224,
    center_crop_size: Optional[int] = None,
    mean: Tuple[float, float, float] = MRI_CHANNEL_MEAN,
    std: Tuple[float, float, float] = MRI_CHANNEL_STD
):

    crop_size = resolve_center_crop_size(target_size, center_crop_size)
    validate_normalization(mean, std)

    return T.Compose([
        T.CenterCrop(crop_size),
        T.Resize(
            size=(target_size, target_size),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        T.Normalize(mean=mean, std=std),
    ])