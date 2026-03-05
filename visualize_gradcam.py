import torch
import matplotlib.pyplot as plt
import numpy as np

from models.model import BrainMRICNN
from src.data.dataloaders import create_dataloaders
from src.utils.gradcam import GradCAM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Load trained model
# -----------------------------
model = BrainMRICNN().to(device)
model.load_state_dict(torch.load("checkpoints/best_model.pth", weights_only=True))
model.eval()


# -----------------------------
# Grad-CAM setup
# -----------------------------
target_layer = model.features[8]
gradcam = GradCAM(model, target_layer)


# -----------------------------
# Load dataset
# -----------------------------
split_path = "data/splits/patient_split.json"

_, _, test_loader = create_dataloaders(
    split_path,
    batch_size=1
)


# ------------------------------------------------
# Find a slice that actually contains brain signal
# ------------------------------------------------
for images, labels, patient_ids in test_loader:

    slice_img = images[0][1].numpy()

    # Skip almost empty slices
    if slice_img.std() > 0.05:
        break


images = images.to(device)


# -----------------------------
# Generate Grad-CAM heatmap
# -----------------------------
cam = gradcam.generate(images)


# -----------------------------
# Prepare image for display
# -----------------------------
image = images[0][1].cpu().numpy()

# Normalize image for visualization
image = (image - image.min()) / (image.max() - image.min() + 1e-8)


# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.title("MRI Slice")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Grad-CAM Heatmap")
plt.imshow(image, cmap="gray")
plt.imshow(cam, cmap="jet", alpha=0.5)
plt.axis("off")

plt.tight_layout()
plt.show()