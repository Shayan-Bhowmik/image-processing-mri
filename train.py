import torch
import torch.nn as nn
import torch.optim as optim

from models.model import BrainMRICNN
from src.data.dataloaders import create_dataloaders


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_path = "data/splits/patient_split.json"

    train_loader, val_loader, test_loader = create_dataloaders(
        split_path,
        batch_size=8
    )

    # Check class distribution
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.tolist())

    print("Total samples:", len(all_labels))
    print("Class 0 count:", all_labels.count(0))
    print("Class 1 count:", all_labels.count(1))

    model = BrainMRICNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 1

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

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Loss: {epoch_loss:.4f}")
        print(f"Accuracy: {epoch_acc:.2f}%")


if __name__ == "__main__":
    train()