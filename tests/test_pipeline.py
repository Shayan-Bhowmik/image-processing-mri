import torch

from src.preprocessing.load_mri import load_nifti
from src.preprocessing.extract_slices import extract_slices
from src.preprocessing.preprocess import normalize, resize_slice
from src.preprocessing.stack_slices import stack_slices
from src.models.cnn_model import CNNModel
from src.explainability.gradcam import GradCAM, overlay


MRI_PATH = r"C:\dataset\brats\BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_flair.nii"


def test_pipeline():

    print("\nSTEP 1: Loading MRI")
    volume = load_nifti(MRI_PATH)
    print("Volume shape:", volume.shape)

    print("\nSTEP 2: Extracting slices")
    slices = extract_slices(volume)
    print("Number of slices:", len(slices))

    print("\nSTEP 3: Preprocessing")
    slice0 = normalize(slices[50])
    slice0 = resize_slice(slice0)
    print("Slice shape:", slice0.shape)

    print("\nSTEP 4: 2.5D stacking")
    stacked = stack_slices(slices)
    print("Stacked slice shape:", stacked[0].shape)

    print("\nSTEP 5: CNN model test")

    from src.models.cnn_model import CNNModel

    model = CNNModel()    

    # create input from real MRI slices
    stacked = stack_slices(slices)

    real_input = stacked[50]

    real_input = torch.tensor(real_input).unsqueeze(0).float()

    x = real_input

    y = model(x)

    print("Model output shape:", y.shape)

    print("\nSTEP 6: GradCAM test")

    import matplotlib.pyplot as plt
    import cv2

    cam = GradCAM(model)

    heatmap = cam.generate(x)

    print("GradCAM heatmap shape:", heatmap.shape)

    # prepare MRI slice
    example_slice = resize_slice(slices[50])

    # resize heatmap to match slice
    heatmap_resized = cv2.resize(heatmap, (224,224))

    # create overlay
    overlay_img = overlay(example_slice, heatmap)

    # display three images
    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.title("MRI Slice")
    plt.imshow(example_slice, cmap="gray")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("GradCAM Heatmap")
    plt.imshow(heatmap_resized, cmap="jet")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title("Overlay")
    plt.imshow(overlay_img)
    plt.axis("off")

    plt.show()

    print("\nPIPELINE TEST SUCCESSFUL")


if __name__ == "__main__":
    test_pipeline()