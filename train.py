import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
import os
from src.utils.seed import set_seed

from models.model import BrainMRICNN
from src.data.dataloaders import create_dataloaders


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_probs = []
    all_patient_ids = []

    with torch.no_grad():
        for images, labels, patient_ids in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            positive_probs = probs[:, 1]

            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(positive_probs.cpu().numpy())
            all_patient_ids.extend(patient_ids)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy, all_labels, all_preds, all_probs, all_patient_ids


def train():
    set_seed(42)

    # =========================
    # CONFIG (NEW)
    # =========================
    config = {
        "use_2_5d": True
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_path = "data/splits/patient_split.json"

    # =========================
    # DATALOADER (UPDATED)
    # =========================
    train_loader, val_loader, test_loader = create_dataloaders(
        split_path,
        batch_size=8,
        use_2_5d=config["use_2_5d"]
    )

    # =========================
    # CLASS WEIGHTS
    # =========================
    all_labels = []
    for _, labels, _ in train_loader:
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

    # =========================
    # MODEL (UPDATED)
    # =========================
    in_channels = 3 if config["use_2_5d"] else 1
    model = BrainMRICNN(in_channels=in_channels).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=1
    )

    num_epochs = 5
    best_val_acc = 0.0

    os.makedirs("checkpoints", exist_ok=True)

    # =========================
    # TRAINING LOOP
    # =========================
    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _ in train_loader:
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

        val_loss, val_acc, _, _, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
        print("-" * 50)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print("✔ Best model saved.")

    # =========================
    # TEST SET EVALUATION
    # =========================
    print("\n===== TEST SET EVALUATION =====")

    model.load_state_dict(
        torch.load("checkpoints/best_model.pth", weights_only=True)
    )
    model.eval()

    test_loss, test_acc, test_true, test_pred, test_probs, test_patient_ids = evaluate(
        model, test_loader, criterion, device
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")

    print("\nTest Confusion Matrix:")
    print(confusion_matrix(test_true, test_pred))

    print("\nTest Classification Report:")
    print(classification_report(test_true, test_pred))

    # =========================
    # SLICE-LEVEL ROC
    # =========================
    print("\nROC-AUC Analysis (Slice-Level):")
    fpr, tpr, _ = roc_curve(test_true, test_probs)
    roc_auc = auc(fpr, tpr)
    print(f"ROC-AUC: {roc_auc:.4f}")

    # =========================
    # PATIENT-LEVEL AGGREGATION
    # =========================
    print("\n===== PATIENT-LEVEL EVALUATION (Max Probability) =====")

    patient_dict = {}

    for pid, label, prob in zip(test_patient_ids, test_true, test_probs):
        if pid not in patient_dict:
            patient_dict[pid] = {
                "true_label": label,
                "slice_probs": []
            }
        patient_dict[pid]["slice_probs"].append(prob)

    patient_labels = []
    max_scores = []

    for pid in patient_dict:
        patient_labels.append(patient_dict[pid]["true_label"])
        max_scores.append(max(patient_dict[pid]["slice_probs"]))

    patient_labels = np.array(patient_labels)
    max_scores = np.array(max_scores)

    patient_preds = (max_scores > 0.5).astype(int)

    print("\nPatient-Level Confusion Matrix:")
    print(confusion_matrix(patient_labels, patient_preds))

    print("\nPatient-Level Classification Report:")
    print(classification_report(patient_labels, patient_preds))

    fpr_p, tpr_p, _ = roc_curve(patient_labels, max_scores)
    roc_auc_patient = auc(fpr_p, tpr_p)
    print(f"\nPatient-Level ROC-AUC: {roc_auc_patient:.4f}")

    # =========================
    # THRESHOLD OPTIMIZATION
    # =========================
    print("\n===== PATIENT-LEVEL THRESHOLD OPTIMIZATION =====")

    thresholds = np.linspace(0, 1, 200)
    best_threshold = 0.5
    best_youden = -1

    for t in thresholds:
        preds = (max_scores > t).astype(int)
        tn, fp, fn, tp = confusion_matrix(patient_labels, preds).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        youden = sensitivity + specificity - 1

        if youden > best_youden:
            best_youden = youden
            best_threshold = t
            best_sensitivity = sensitivity
            best_specificity = specificity

    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"Sensitivity at Best Threshold: {best_sensitivity:.4f}")
    print(f"Specificity at Best Threshold: {best_specificity:.4f}")


if __name__ == "__main__":
    train()