import torch
import numpy as np
import cv2


class GradCAM:

    def __init__(self, model):

        self.model = model
        self.model.eval()

    def generate(self, input_tensor):
        """
        Generate GradCAM heatmap
        """

        output = self.model(input_tensor)

        class_idx = torch.argmax(output)

        self.model.zero_grad()

        output[0, class_idx].backward()

        gradients = self.model.gradients
        activations = self.model.activations

        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        for i in range(activations.shape[1]):

            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()

        heatmap = heatmap.detach().cpu().numpy()

        heatmap = np.maximum(heatmap, 0)

        heatmap = heatmap / np.max(heatmap)

        return heatmap


def brain_mask(slice_img):
    """
    Remove skull edges from GradCAM
    """

    mask = slice_img > 0

    return mask.astype(float)


def overlay(slice_img, heatmap):

    import cv2
    import numpy as np

    # Resize heatmap to match slice
    heatmap = cv2.resize(heatmap, (slice_img.shape[1], slice_img.shape[0]))

    # Normalize heatmap
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / (heatmap.max() + 1e-8)

    # Convert heatmap to color
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # Convert slice to RGB
    if len(slice_img.shape) == 2:
        slice_img = cv2.cvtColor(np.uint8(slice_img), cv2.COLOR_GRAY2RGB)

    slice_img = cv2.resize(slice_img, (heatmap.shape[1], heatmap.shape[0]))

    # Blend images
    overlay_img = cv2.addWeighted(slice_img, 0.6, heatmap, 0.4, 0)

    return overlay_img