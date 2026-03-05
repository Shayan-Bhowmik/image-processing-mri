import torch
import matplotlib.pyplot as plt
import numpy as np
import os

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


# -----------------------------
# Output folder
# -----------------------------
output_dir = "results/gradcam"
os.makedirs(output_dir, exist_ok=True)


saved_images = 0
max_images = 5


for images, labels, patient_ids in test_loader:

    slice_img = images[0][1].numpy()

    # Skip nearly empty slices
    if slice_img.std() < 0.05:
        continue

    images = images.to(device)

    # Generate heatmap
    cam = gradcam.generate(images)

    # Normalize MRI slice for visualization
    image = images[0][1].cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

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

    save_path = os.path.join(output_dir, f"gradcam_{saved_images+1}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"Saved {save_path}")

    saved_images += 1

    if saved_images >= max_images:
        break


print("\nGrad-CAM generation complete.")