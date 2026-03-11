import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
import cv2
from torchvision import models

# Step 6.1 — Load Trained Model

def load_model(model_path):

    model = models.resnet50(weights=None)

    # Change final layer for 2 classes
    model.fc = nn.Linear(model.fc.in_features, 2)

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    # Set evaluation mode
    model.eval()

    return model

# Step 6.2 — Load MRI Volume

def load_mri(mri_path):

    img = nib.load(mri_path)
    volume = img.get_fdata()

    return volume

# Generate 2.5D slices (i-1, i, i+1)

def generate_slices(volume):

    slices = []

    for i in range(1, volume.shape[2] - 1):

        slice_25d = np.stack([
            volume[:, :, i - 1],
            volume[:, :, i],
            volume[:, :, i + 1]
        ], axis=0)

        slices.append(slice_25d)

    return slices

# Preprocess slice for model

def preprocess_slice(slice_img):

    # Convert (3,H,W) -> (H,W,3)
    slice_img = np.transpose(slice_img, (1, 2, 0))

    # Resize to 224x224
    slice_img = cv2.resize(slice_img, (224, 224))

    # Normalize
    if np.max(slice_img) != 0:
        slice_img = slice_img / np.max(slice_img)

    # Convert to tensor
    slice_img = torch.tensor(slice_img).permute(2, 0, 1).float()

    return slice_img

# Full prediction pipeline

def predict_mri(model, mri_path):

    volume = load_mri(mri_path)

    slices = generate_slices(volume)

    predictions = []

    for s in slices:

        processed = preprocess_slice(s)

        prob = predict_slice(model, processed)

        predictions.append(prob)

    final_prediction = aggregate_predictions(predictions)

    return final_prediction

# Main

if __name__ == "__main__":

    MODEL_PATH = "models/mri_resnet50.pth"

    MRI_PATH = r"C:\brats\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_flair.nii"

    model = load_model(MODEL_PATH)

    result = predict_mri(model, MRI_PATH)

    normal_prob = result[0].item()
    tumor_prob = result[1].item()

    print("\nPrediction Probabilities:")
    print(f"Normal : {normal_prob:.4f}")
    print(f"Tumor  : {tumor_prob:.4f}")

    if tumor_prob > normal_prob:
        print("\nFinal Prediction: Tumor Detected")
    else:
        print("\nFinal Prediction: Normal MRI")