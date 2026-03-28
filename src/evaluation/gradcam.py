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
            class_idx = int(torch.argmax(output, dim=1).item())

        loss = output[0, class_idx]
        loss.backward()

        
        gradients = self.model.get_gradients()[0]
        activations = self.model.get_conv_features()[0]

        
        gradients = gradients / (torch.mean(torch.abs(gradients)) + 1e-8)
        weights = torch.mean(gradients, dim=(1, 2))
        
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32).to(activations.device)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        
        cam = torch.relu(cam)

        
        cam = cam.detach().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)  
        

        
        cam = cv2.resize(cam, (224, 224))

        return cam


def save_gradcam_panel(image, cam, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    base, ext = os.path.splitext(save_path)

    
    counter = 1
    new_path = f"{base}_{counter}{ext}"
    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base}_{counter}{ext}"

    
    image = image - image.min()
    image = image / (image.max() + 1e-8)
    image = np.uint8(255 * image)

    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    
    overlay = cv2.addWeighted(image_rgb, 0.7, heatmap, 0.3, 0)

    
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

    print(f"\nGrad-CAM saved at {new_path}")