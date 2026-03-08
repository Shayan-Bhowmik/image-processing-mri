import torch
import torch.nn as nn
from torchvision import models

def build_model():

    # Load pretrained ResNet50
    model = models.resnet50(pretrained=True)

    # Modify final layer for 2 classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    return model