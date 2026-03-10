import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import cv2
import matplotlib.pyplot as plt


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):

        def forward_hook(module, input, output):
            self.feature_maps = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor):

        output = self.model(input_tensor)
        class_idx = torch.argmax(output)

        self.model.zero_grad()
        output[0, class_idx].backward()

        gradients = self.gradients
        feature_maps = self.feature_maps

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * feature_maps, dim=1)

        cam = F.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


def generate_dummy_mri():
    """Create a fake MRI-like image"""
    img = np.zeros((224,224,3), dtype=np.uint8)

    cv2.circle(img,(112,112),80,(120,120,120),-1)
    cv2.circle(img,(140,100),25,(200,200,200),-1)  # fake tumor

    return img


def overlay_heatmap(original_image, heatmap):

    heatmap = cv2.resize(heatmap,(224,224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(original_image,0.6,heatmap,0.4,0)

    return overlay


if __name__ == "__main__":

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(2048,2)
    model.eval()

    gradcam = GradCAM(model, model.layer4)

    dummy_image = generate_dummy_mri()

    input_tensor = torch.tensor(dummy_image/255.).permute(2,0,1).unsqueeze(0).float()

    cam = gradcam.generate_cam(input_tensor)

    overlay = overlay_heatmap(dummy_image, cam)

    plt.imshow(overlay)
    plt.title("Grad-CAM Example")
    plt.axis("off")
    plt.show()