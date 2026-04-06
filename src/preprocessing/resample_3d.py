import numpy as np
import torch
import torch.nn.functional as F


def resample_volume_3d(volume: np.ndarray, target_shape=(192, 192, 160)) -> np.ndarray:
    """Resample a 3D volume to a canonical shape using trilinear interpolation."""
    volume = np.asarray(volume, dtype=np.float32)

    if tuple(volume.shape) == tuple(target_shape):
        return volume

    tensor = torch.as_tensor(volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(
        tensor,
        size=target_shape,
        mode="trilinear",
        align_corners=False,
    )

    return resized.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32, copy=False)
