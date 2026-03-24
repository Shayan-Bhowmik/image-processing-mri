import torch
import numpy as np
import cv2
import os


class GradCAM:
    def __init__(self, model):
        self.model = model

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output)

        loss = output[0, class_idx]
        loss.backward()

        gradients = self.model.get_gradients()[0]
        activations = self.model.get_conv_features()[0]

        grad_scale = torch.sqrt(torch.mean(gradients ** 2, dim=(1, 2), keepdim=True) + 1e-8)
        gradients = gradients / grad_scale

        clip_value = torch.quantile(torch.abs(gradients).reshape(-1), 0.99)
        gradients = torch.clamp(gradients, -clip_value, clip_value)

        activations = torch.relu(activations)
        act_max = torch.amax(activations, dim=(1, 2), keepdim=True) + 1e-8
        activations = activations / act_max

        weights = torch.mean(torch.relu(gradients), dim=(1, 2))

        cam = torch.sum(weights[:, None, None] * activations, dim=0)
        cam = torch.relu(cam)
        cam = cam.detach().cpu().numpy()

        cam = cam - cam.min()
        scale = np.percentile(cam, 99)
        cam = cam / (scale + 1e-8)
        cam = np.clip(cam, 0.0, 1.0)

        cam = np.power(cam, 0.7)

        cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)

        return cam


def save_gradcam_panel(image, cam, save_path):

    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    base, ext = os.path.splitext(save_path)

    counter = 1
    new_path = f"{base}_{counter}{ext}"

    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base}_{counter}{ext}"

    
    image = image - image.min()
    image = image / (image.max() + 1e-8)

    mask = image > 0
    if np.any(mask):
        ys, xs = np.where(mask)
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()

        pad_y = max(2, int(0.05 * (y2 - y1 + 1)))
        pad_x = max(2, int(0.05 * (x2 - x1 + 1)))

        y1 = max(0, y1 - pad_y)
        y2 = min(image.shape[0] - 1, y2 + pad_y)
        x1 = max(0, x1 - pad_x)
        x2 = min(image.shape[1] - 1, x2 + pad_x)

        image = image[y1:y2 + 1, x1:x2 + 1]
        cam = cam[y1:y2 + 1, x1:x2 + 1]

    image = np.uint8(255 * image)

    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    
    overlay = cv2.addWeighted(image_rgb, 0.6, heatmap, 0.4, 0)

    
    image_rgb = cv2.resize(image_rgb, (224, 224))
    heatmap = cv2.resize(heatmap, (224, 224))
    overlay = cv2.resize(overlay, (224, 224))

    
    def add_title(img, text):
        img = img.copy()
        cv2.putText(img, text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
        return img

    image_rgb = add_title(image_rgb, "MRI Slice")
    heatmap = add_title(heatmap, "Grad-CAM")
    overlay = add_title(overlay, "Overlay")

    
    panel = np.hstack([image_rgb, heatmap, overlay])

    
    cv2.imwrite(new_path, panel)

    print(f"\nGrad-CAM panel saved at {new_path}")