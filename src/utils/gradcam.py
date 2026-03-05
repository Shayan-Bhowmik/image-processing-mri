import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):

        def forward_hook(module, input, output):
            # Save activations
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # Save gradients
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):

        self.model.zero_grad()

        output = self.model(input_tensor)

        # Select predicted class if none provided
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        loss = output[:, class_idx]
        loss.backward()

        gradients = self.gradients
        activations = self.activations

        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination
        cam = (weights * activations).sum(dim=1, keepdim=True)

        # ReLU to keep positive influence
        cam = F.relu(cam)

        # Resize CAM to input size
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze().cpu().numpy()

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Keep only strongest activations (tumor focus)
        threshold = np.percentile(cam, 85)
        cam[cam < threshold] = 0

        # Smooth heatmap
        cam = gaussian_filter(cam, sigma=3)

        return cam