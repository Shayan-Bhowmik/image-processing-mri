import os
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from models.model import BrainMRICNN, FlexibleMultiModalBrainMRI
from src.preprocessing.load_nifti import load_nifti
from src.preprocessing.normalize import zscore_normalize
from src.preprocessing.resample_3d import resample_volume_3d
from src.preprocessing.slice_extraction import extract_valid_slices
from src.preprocessing.resize import resize_sample
from src.utils.gradcam import GradCAM


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _strip_module_prefix(state_dict: dict) -> dict:
    """Handle checkpoints saved with DataParallel by removing leading 'module.'."""
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if not first_key.startswith("module."):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def get_model_input_channels(model: nn.Module) -> int:
    """Return expected input channels for preprocessing."""
    if isinstance(model, FlexibleMultiModalBrainMRI):
        return int(model.num_modalities)
    if isinstance(model, BrainMRICNN):
        return int(model.features[0].in_channels)
    return 3


def load_trained_model(
    checkpoint_path: str = "checkpoints/best_model.pth",
    in_channels: int = 3,
    num_classes: int = 2,
) -> Tuple[nn.Module, torch.device]:
    """Load trained model and return (model, device)."""
    device = get_device()

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = _strip_module_prefix(state_dict)

    if "adaptive_conv.weight" in state_dict and "modality_weights" in state_dict:
        detected_modalities = int(state_dict["modality_weights"].shape[0])
        detected_classes = int(state_dict["classifier.4.weight"].shape[0])
        model = FlexibleMultiModalBrainMRI(
            num_classes=detected_classes,
            num_modalities=detected_modalities,
            modality_dropout_rate=0.0,
        ).to(device)
    else:
        detected_in_channels = int(state_dict.get("features.0.weight").shape[1]) if "features.0.weight" in state_dict else int(in_channels)
        detected_classes = int(state_dict.get("classifier.4.weight").shape[0]) if "classifier.4.weight" in state_dict else int(num_classes)
        model = BrainMRICNN(num_classes=detected_classes, in_channels=detected_in_channels).to(device)

    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def _stack_25d_from_valid_slices(valid_slices: List[np.ndarray]) -> List[np.ndarray]:
    samples = []
    n = len(valid_slices)

    for i in range(n):
        prev_slice = valid_slices[i - 1] if i > 0 else valid_slices[i]
        curr_slice = valid_slices[i]
        next_slice = valid_slices[i + 1] if i < n - 1 else valid_slices[i]
        samples.append(np.stack([prev_slice, curr_slice, next_slice], axis=0))

    return samples


def _build_model_input_samples(
    valid_slices: List[np.ndarray],
    model_in_channels: int,
    single_modality_name: str = "t1",
) -> List[np.ndarray]:
    """
    Build per-slice model inputs for the expected channel count.

    - 3 channels: classic 2.5D (prev/current/next)
    - Other channel counts: replicate current slice across channels
    """
    if model_in_channels == 3:
        return _stack_25d_from_valid_slices(valid_slices)

    if model_in_channels == 4:
        # Keep channel semantics aligned with training order in dataset loader.
        # Default order: [flair, t1, t1c, t2]
        modality_order = ["flair", "t1", "t1c", "t2"]
        selected_modality = str(single_modality_name).lower().strip()
        selected_index = modality_order.index(selected_modality) if selected_modality in modality_order else 1

        samples = []
        for curr_slice in valid_slices:
            sample = np.zeros((model_in_channels, curr_slice.shape[0], curr_slice.shape[1]), dtype=curr_slice.dtype)
            sample[selected_index] = curr_slice
            samples.append(sample)
        return samples

    samples = []
    for curr_slice in valid_slices:
        samples.append(np.repeat(curr_slice[None, ...], model_in_channels, axis=0))
    return samples


def preprocess_volume(
    volume: np.ndarray,
    image_size: Tuple[int, int] = (224, 224),
    canonical_shape: Tuple[int, int, int] = (192, 192, 160),
    fixed_slice_count: int = 96,
    model_in_channels: int = 3,
    single_modality_name: str = "t1",
) -> Dict[str, object]:
    """Preprocess 3D MRI volume and build per-slice 2.5D tensors."""
    if len(volume.shape) == 4:
        volume = volume[..., 0]

    volume = volume.astype(np.float32, copy=False)
    volume = resample_volume_3d(volume, target_shape=canonical_shape)
    volume = zscore_normalize(volume)

    all_valid_slices = extract_valid_slices(volume, fixed_count=None)
    valid_slices = extract_valid_slices(volume, fixed_count=fixed_slice_count)
    if not valid_slices:
        raise ValueError("No valid slices found after filtering. Try a different scan.")

    stacked_samples = _build_model_input_samples(
        valid_slices,
        model_in_channels=model_in_channels,
        single_modality_name=single_modality_name,
    )

    tensors = []
    for sample in stacked_samples:
        tensors.append(resize_sample(sample, size=image_size))

    input_batch = torch.stack(tensors, dim=0).float()

    return {
        "valid_slices": valid_slices,
        "total_valid_slices": len(all_valid_slices),
        "used_valid_slices": len(valid_slices),
        "input_batch": input_batch,
    }


def preprocess_uploaded_nifti(
    uploaded_bytes: bytes,
    uploaded_filename: str | None = None,
    image_size: Tuple[int, int] = (224, 224),
    canonical_shape: Tuple[int, int, int] = (192, 192, 160),
    fixed_slice_count: int = 96,
    model_in_channels: int = 3,
    single_modality_name: str = "t1",
) -> Dict[str, object]:
    """Load uploaded NIfTI bytes and return preprocessed tensors and source slices."""
    suffix = ".nii.gz"
    if uploaded_filename and uploaded_filename.lower().endswith(".nii"):
        suffix = ".nii"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_bytes)
        tmp_path = tmp.name

    try:
        volume = load_nifti(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return preprocess_volume(
        volume,
        image_size=image_size,
        canonical_shape=canonical_shape,
        fixed_slice_count=fixed_slice_count,
        model_in_channels=model_in_channels,
        single_modality_name=single_modality_name,
    )


def preprocess_uploaded_multimodal_nifti(
    uploaded_by_modality: Dict[str, Tuple[bytes, str]],
    image_size: Tuple[int, int] = (224, 224),
    canonical_shape: Tuple[int, int, int] = (192, 192, 160),
    fixed_slice_count: int = 96,
    modality_order: Tuple[str, ...] = ("flair", "t1", "t1c", "t2"),
) -> Dict[str, object]:
    """
    Build a true multi-modal input batch from uploaded NIfTI files.

    Args:
        uploaded_by_modality: mapping modality -> (bytes, original_filename)
            Expected keys: flair, t1, t1c, t2 (missing keys are zero-filled).
    """
    modality_volumes: Dict[str, np.ndarray] = {}

    for modality in modality_order:
        if modality not in uploaded_by_modality:
            continue

        uploaded_bytes, uploaded_filename = uploaded_by_modality[modality]
        suffix = ".nii.gz"
        if uploaded_filename and uploaded_filename.lower().endswith(".nii"):
            suffix = ".nii"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded_bytes)
            tmp_path = tmp.name

        try:
            volume = load_nifti(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        if len(volume.shape) == 4:
            volume = volume[..., 0]

        volume = volume.astype(np.float32, copy=False)
        volume = resample_volume_3d(volume, target_shape=canonical_shape)
        volume = zscore_normalize(volume)
        modality_volumes[modality] = volume

    if not modality_volumes:
        raise ValueError("No valid modality files were provided.")

    reference_modality = "flair" if "flair" in modality_volumes else next(iter(modality_volumes.keys()))
    reference_volume = modality_volumes[reference_modality]
    all_valid_slices_ref = extract_valid_slices(reference_volume, fixed_count=None)
    valid_slices_ref = extract_valid_slices(reference_volume, fixed_count=fixed_slice_count)
    if not valid_slices_ref:
        raise ValueError("No valid slices found after filtering. Try different scan files.")

    modality_slices_map: Dict[str, List[np.ndarray]] = {}
    for modality, volume in modality_volumes.items():
        slices = extract_valid_slices(volume, fixed_count=fixed_slice_count)
        if len(slices) != len(valid_slices_ref):
            slices = valid_slices_ref
        modality_slices_map[modality] = slices

    tensors = []
    for idx in range(len(valid_slices_ref)):
        channel_slices = []
        for modality in modality_order:
            if modality in modality_slices_map:
                channel_slices.append(modality_slices_map[modality][idx])
            else:
                channel_slices.append(np.zeros_like(valid_slices_ref[idx]))

        sample = np.stack(channel_slices, axis=0)
        tensors.append(resize_sample(sample, size=image_size))

    input_batch = torch.stack(tensors, dim=0).float()

    return {
        "valid_slices": valid_slices_ref,
        "total_valid_slices": len(all_valid_slices_ref),
        "used_valid_slices": len(valid_slices_ref),
        "input_batch": input_batch,
    }


def predict_slices(
    model: nn.Module,
    input_batch: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (class_predictions, positive_class_probabilities) for each slice."""
    with torch.no_grad():
        logits = model(input_batch.to(device))
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)

    return pred.cpu().numpy(), probs[:, 1].cpu().numpy()


def aggregate_patient_score(slice_probs: np.ndarray, top_k: int = 10) -> float:
    """Aggregate slice probabilities into one patient score using top-k mean."""
    if slice_probs.size == 0:
        raise ValueError("Slice probabilities are empty.")

    k = min(top_k, slice_probs.size)
    top_values = np.sort(slice_probs)[-k:]
    return float(np.mean(top_values))


def build_gradcam_for_slice(
    model: nn.Module,
    device: torch.device,
    sample_tensor: torch.Tensor,
    target_class: int | None = None,
    smooth_kernel: int = 5,
    clip_percentiles: Tuple[float, float] = (2.0, 99.5),
    apply_brain_mask: bool = True,
    brain_mask_threshold: float = 0.05,
) -> np.ndarray:
    """Generate Grad-CAM heatmap for one preprocessed sample tensor (3, H, W)."""

    gradcam = GradCAM(model, model.features[8])

    input_tensor = sample_tensor.unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    try:
        heatmap = gradcam.generate(
            input_tensor,
            class_idx=target_class,
            smooth_kernel=smooth_kernel,
            clip_percentiles=clip_percentiles,
        )
    finally:
        gradcam.remove_hooks()

    if apply_brain_mask:
        sample_np = sample_tensor.detach().cpu().numpy()

        # Use the channel with strongest foreground signal for masking.
        # This prevents blank masks when single-modality uploads are placed in
        # channels other than index 1 (e.g., FLAIR/T1c/T2 in 4-channel mode).
        if sample_np.ndim == 3 and sample_np.shape[0] > 1:
            channel_signal = [float(np.count_nonzero(np.abs(sample_np[c]) > 1e-8)) for c in range(sample_np.shape[0])]
            best_channel_idx = int(np.argmax(channel_signal))
            base_slice = sample_np[best_channel_idx]
        elif sample_np.ndim == 3:
            base_slice = sample_np[0]
        else:
            base_slice = sample_np

        base_slice = (base_slice - base_slice.min()) / (base_slice.max() - base_slice.min() + 1e-8)
        brain_mask = (base_slice > brain_mask_threshold).astype(np.float32)

        heatmap = heatmap * brain_mask
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / (heatmap.max() + 1e-8)

    return np.clip(heatmap, 0.0, 1.0)
