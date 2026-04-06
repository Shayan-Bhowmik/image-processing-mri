import torch
import numpy as np

from src.models.cnn_model import CNNModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path="models/best_model.pth"):
    """
    Load trained CNN model
    """

    model = CNNModel()

    model.load_state_dict(torch.load(model_path, map_location=device))

    model = model.to(device)

    model.eval()

    return model


def predict_slice(model, slice_tensor):
    """
    Predict tumor probability for one slice
    """

    with torch.no_grad():

        slice_tensor = slice_tensor.to(device)

        output = model(slice_tensor)

        probs = torch.softmax(output, dim=1)

    return probs.cpu().numpy()