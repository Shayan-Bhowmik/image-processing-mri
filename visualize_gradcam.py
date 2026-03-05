import torch
import matplotlib.pyplot as plt
import numpy as np

from models.model import BrainMRICNN
from src.data.dataloaders import create_dataloaders
from src.utils.gradcam import GradCAM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load trained model
model = BrainMRICNN().to(device)
model.load_state_dict(torch.load("checkpoints/best_model.pth", weights_only=True))
model.eval()


# Target convolution layer for Grad-CAM
target_layer = model.features[8]

gradcam = GradCAM(model, target_layer)


# Load dataset
split_path = "data/splits/patient_split.json"

_, _, test_loader = create_dataloaders(
    split_path,
    batch_size=1
)


# Get one test sample
images, labels, patient_ids = next(iter(test_loader))

images = images.to(device)


# Generate heatmap
cam = gradcam.generate(images)


# Convert MRI slice for visualization
image = images[0][1].cpu().numpy()


plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.title("MRI Slice")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Grad-CAM")
plt.imshow(image, cmap="gray")
plt.imshow(cam, cmap="jet", alpha=0.5)
plt.axis("off")

plt.show()