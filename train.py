import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from models.model import BrainMRICNN
from src.data.dataloaders import create_dataloaders


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_path = "data/splits/patient_split.json"

    train_loader, val_loader, test_loader = create_dataloaders(
        split_path,
        batch_size=8
    )

    # -------------------------------
    # Step 6.1 — Class Weights
    # -------------------------------
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.tolist())

    print("Total samples:", len(all_labels))
    print("Class 0 count:", all_labels.count(0))
    print("Class 1 count:", all_labels.count(1))

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(all_labels),
        y=all_labels
    )

    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print("Class Weights:", class_weights)

    # -------------------------------
    # Model + Loss + Optimizer
    # -------------------------------
    model = BrainMRICNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
        print("-" * 50)


if __name__ == "__main__":
    train()