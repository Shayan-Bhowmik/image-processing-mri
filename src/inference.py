import os
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import torch

from models.model import BrainMRICNN
from src.preprocessing.load_nifti import load_nifti
from src.preprocessing.normalize import zscore_normalize
from src.preprocessing.slice_extraction import extract_valid_slices
from src.preprocessing.resize import resize_sample
from src.utils.gradcam import GradCAM


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_model(
    checkpoint_path: str = "checkpoints/best_model.pth",
    in_channels: int = 3,
    num_classes: int = 2,
) -> Tuple[BrainMRICNN, torch.device]:
    """Load trained model and return (model, device)."""
    device = get_device()
    model = BrainMRICNN(num_classes=num_classes, in_channels=in_channels).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
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


def preprocess_volume(volume: np.ndarray, image_size: Tuple[int, int] = (224, 224)) -> Dict[str, object]:
    """Preprocess 3D MRI volume and build per-slice 2.5D tensors."""
    if len(volume.shape) == 4:
        volume = volume[..., 0]

    volume = volume.astype(np.float32, copy=False)
    volume = zscore_normalize(volume)

    valid_slices = extract_valid_slices(volume)
    if not valid_slices:
        raise ValueError("No valid slices found after filtering. Try a different scan.")

    stacked_samples = _stack_25d_from_valid_slices(valid_slices)

    tensors = []
    for sample in stacked_samples:
        tensors.append(resize_sample(sample, size=image_size))

    input_batch = torch.stack(tensors, dim=0).float()

    return {
        "valid_slices": valid_slices,
        "input_batch": input_batch,
    }


def preprocess_uploaded_nifti(
    uploaded_bytes: bytes,
    uploaded_filename: str | None = None,
    image_size: Tuple[int, int] = (224, 224),
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

    return preprocess_volume(volume, image_size=image_size)


def predict_slices(
    model: BrainMRICNN,
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
    model: BrainMRICNN,
    device: torch.device,
    sample_tensor: torch.Tensor,
    target_class: int | None = None,
    smooth_kernel: int = 5,
    clip_percentiles: Tuple[float, float] = (2.0, 99.5),
    apply_brain_mask: bool = True,
    brain_mask_threshold: float = 0.05,
) -> np.ndarray:
    """Generate Grad-CAM heatmap for one preprocessed sample tensor (3, H, W)."""
    # Last convolution layer usually gives more class-discriminative maps.
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
        base_slice = sample_tensor[1].detach().cpu().numpy()
        base_slice = (base_slice - base_slice.min()) / (base_slice.max() - base_slice.min() + 1e-8)
        brain_mask = (base_slice > brain_mask_threshold).astype(np.float32)

        heatmap = heatmap * brain_mask
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / (heatmap.max() + 1e-8)

    return np.clip(heatmap, 0.0, 1.0)
