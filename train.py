import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

from models.model import BrainMRICNN
from src.data.dataloaders import create_dataloaders


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy, all_labels, all_preds


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------
    # SANITY TEST FLAG
    # -------------------------------
    SANITY_LABEL_SHUFFLE = True  # Turn OFF after sanity test

    split_path = "data/splits/patient_split.json"

    train_loader, val_loader, test_loader = create_dataloaders(
        split_path,
        batch_size=8
    )

    # -------------------------------
    # Class Weights
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
    # Model
    # -------------------------------
    model = BrainMRICNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Reduced epochs for sanity test
    num_epochs = 2

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # -------------------------------
            # Shuffle labels (Sanity Test Only)
            # -------------------------------
            if SANITY_LABEL_SHUFFLE:
                perm = torch.randperm(labels.size(0))
                labels = labels[perm]

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

        val_loss, val_acc, val_true, val_pred = evaluate(
            model, val_loader, criterion, device
        )

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
        print("-" * 50)

    # -------------------------------
    # Final Confusion Matrix
    # -------------------------------
    print("\nValidation Confusion Matrix:")
    cm = confusion_matrix(val_true, val_pred)
    print(cm)

    print("\nClassification Report:")
    print(classification_report(val_true, val_pred))


if __name__ == "__main__":
    train()