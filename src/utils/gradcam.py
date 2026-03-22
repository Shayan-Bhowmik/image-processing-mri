import torch
import torch.nn.functional as F
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):

        def forward_hook(module, input, output):
            # Store activations (detach to avoid graph issues)
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # Store gradients (detach to avoid accumulation issues)
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=1):

        # -----------------------------
        # Reset gradients
        # -----------------------------
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)

        # -----------------------------
        # FORCE TARGET CLASS
        # -----------------------------
        target = output[:, class_idx]

        # Backward pass
        target.backward(torch.ones_like(target))

        # -----------------------------
        # Safety checks (IMPORTANT)
        # -----------------------------
        if self.gradients is None or self.activations is None:
            raise RuntimeError("GradCAM hooks not working properly.")

        gradients = self.gradients
        activations = self.activations

        # -----------------------------
        # Compute weights
        # -----------------------------
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # -----------------------------
        # Compute CAM
        # -----------------------------
        cam = (weights * activations).sum(dim=1, keepdim=True)

        cam = F.relu(cam)

        # -----------------------------
        # Resize to input size
        # -----------------------------
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze().cpu().numpy()

        # -----------------------------
        # Normalize safely
        # -----------------------------
        cam = cam - cam.min()

        if cam.max() != 0:
            cam = cam / cam.max()

        return cam