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
max_images = 15
fallback_samples = []

max_iterations = 200
iteration = 0


for images, labels, patient_ids in test_loader:

    iteration += 1
    if iteration > max_iterations:
        print("Stopping early")
        break

    slice_img = images[0][1].numpy()

    if np.sum(slice_img > 0) < 5000:
        continue

    if slice_img.std() < 0.05:
        continue

    # -----------------------------
    # NORMALIZATION
    # -----------------------------
    images = images.to(device).float()
    images.requires_grad_(True)

    # -----------------------------
    # Forward
    # -----------------------------
    outputs = model(images)
    probs = torch.softmax(outputs, dim=1)
    tumor_prob = probs[0, 1].item()

    if iteration % 10 == 0:
        print(f"Prob: {tumor_prob:.4f}")

    # -----------------------------
    # Grad-CAM
    # -----------------------------
    cam = gradcam.generate(
        images,
        class_idx=1,
        smooth_kernel=5,
        clip_percentiles=(2.0, 99.5),
    )

    fallback_samples.append((images.detach().clone(), tumor_prob, cam))

    # -----------------------------
    # Prepare image
    # -----------------------------
    image = images[0][1].detach().cpu().numpy()
    image = (image - image.min()) / (image.max() + 1e-8)

    # -----------------------------
    # CLEAN CAM (ONLY THIS)
    # -----------------------------
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    print("CAM range:", cam.min(), cam.max())

    heatmap = np.clip(cam, 0, 1)
    heatmap_color = plt.cm.jet(heatmap)[:, :, :3]

    # -----------------------------
    # Overlay
    # -----------------------------
    overlay = image[..., None] * 0.4 + heatmap_color * 0.6


    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("MRI Slice")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Grad-CAM Heatmap")
    plt.imshow(heatmap, cmap="jet")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    save_path = os.path.join(output_dir, f"gradcam_{saved_images+1}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"Saved {save_path}")

    saved_images += 1

    if saved_images >= max_images:
        break


# -----------------------------
# Fallback
# -----------------------------
if saved_images < max_images:

    print(f"\nUsing fallback...")

    fallback_samples.sort(key=lambda x: x[1], reverse=True)

    for images, prob, cam in fallback_samples:

        if saved_images >= max_images:
            break

        image = images[0][1].detach().cpu().numpy()
        image = (image - image.min()) / (image.max() + 1e-8)

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        heatmap = np.clip(cam, 0, 1)
        heatmap_color = plt.cm.jet(heatmap)[:, :, :3]

        overlay = image[..., None] * 0.4 + heatmap_color * 0.6

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.title("Fallback MRI")
        plt.imshow(image, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Grad-CAM Heatmap")
        plt.imshow(heatmap, cmap="jet")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Overlay")
        plt.imshow(overlay)
        plt.axis("off")

        save_path = os.path.join(output_dir, f"gradcam_{saved_images+1}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        print(f"Saved fallback {save_path}")

        saved_images += 1


print("\nGrad-CAM generation complete.")