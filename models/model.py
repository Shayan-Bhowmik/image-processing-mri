import torch
import torch.nn as nn
import torch.nn.functional as F


class BrainMRICNN(nn.Module):
    def __init__(self, num_classes=2, in_channels=3):
        super(BrainMRICNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class MultiModalBrainMRI(nn.Module):
    """
    Multi-modal MRI model with modality-weighted fusion.

    Learns importance weights for each of the 4 MRI modalities (T1, T1c, T2, FLAIR)
    and uses weighted fusion to combine them before feature extraction.

    Input: (batch, 4, 224, 224) - 4 channels representing [T1, T1c, T2, FLAIR]
    Output: (batch, num_classes) - Classification logits
    """

    def __init__(self, num_classes=2, num_modalities=4):
        super(MultiModalBrainMRI, self).__init__()

        self.modality_weights = nn.Parameter(torch.ones(num_modalities) / num_modalities)
        self.num_modalities = num_modalities

        self.features = nn.Sequential(
            nn.Conv2d(num_modalities, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        weighted_x = x * self.modality_weights.view(1, self.num_modalities, 1, 1)
        x = self.features(weighted_x)
        x = self.classifier(x)
        return x

    def get_modality_weights(self):
        modality_names = ["T1", "T1c", "T2", "FLAIR"]
        weights = self.modality_weights.detach().cpu().numpy()
        return {name: float(w) for name, w in zip(modality_names, weights)}


class FlexibleMultiModalBrainMRI(nn.Module):
    """
    Flexible multi-modal MRI model that works with variable number of modalities.
    """

    def __init__(self, num_classes=2, num_modalities=4, modality_dropout_rate=0.0):
        super(FlexibleMultiModalBrainMRI, self).__init__()

        self.num_modalities = num_modalities
        self.modality_dropout_rate = modality_dropout_rate
        self.modality_weights = nn.Parameter(torch.ones(num_modalities) / num_modalities)

        self.adaptive_conv = nn.Conv2d(num_modalities, 4, kernel_size=1, padding=0)

        self.features = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, apply_modality_dropout=False):
        batch_size = x.size(0)

        if self.training and apply_modality_dropout and self.modality_dropout_rate > 0:
            mask = torch.bernoulli(
                torch.full((batch_size, self.num_modalities, 1, 1), 1.0 - self.modality_dropout_rate)
            ).to(x.device)
            x = x * mask

        weighted_x = x * self.modality_weights.view(1, self.num_modalities, 1, 1)
        x = self.adaptive_conv(weighted_x)
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_modality_weights(self, modality_names=None):
        if modality_names is None:
            modality_names = [f"Modality_{i}" for i in range(self.num_modalities)]

        weights = self.modality_weights.detach().cpu().numpy()
        return {name: float(w) for name, w in zip(modality_names, weights)}