import torch
import torch.nn.functional as F


def resize_sample(sample, size=(224, 224)):
    """
    Resize a 2.5D sample from (3, H, W) to (3, 224, 224)

    Parameters:
        sample (np.ndarray): Shape (3, H, W)
        size (tuple): Target size (height, width)

    Returns:
        torch.Tensor: Resized tensor (3, 224, 224)
    """
    tensor = torch.as_tensor(sample, dtype=torch.float32).unsqueeze(0)

    resized = F.interpolate(
        tensor,
        size=size,
        mode="bilinear",
        align_corners=False
    )

    return resized.squeeze(0)  # (3, 224, 224)