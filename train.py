import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
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


def evaluate_patient_level(
    slice_labels,
    slice_probs,
    slice_patient_ids,
    threshold=0.5,
    top_k=10,
):
    """Aggregate slice outputs per patient using top-k mean (same rule as inference)."""
    patient_prob_map = defaultdict(list)
    patient_label_map = {}

    for label, prob, patient_id in zip(slice_labels, slice_probs, slice_patient_ids):
        patient_prob_map[patient_id].append(float(prob))
        if patient_id not in patient_label_map:
            patient_label_map[patient_id] = int(label)

    patient_true = []
    patient_pred = []
    patient_scores = []

    for patient_id, probs in patient_prob_map.items():
        probs_arr = np.asarray(probs, dtype=np.float32)
        k = min(top_k, probs_arr.size)
        patient_score = float(np.mean(np.sort(probs_arr)[-k:]))

        true_label = patient_label_map[patient_id]
        pred_label = 1 if patient_score >= threshold else 0

        patient_true.append(true_label)
        patient_pred.append(pred_label)
        patient_scores.append(patient_score)

    patient_true = np.asarray(patient_true, dtype=np.int32)
    patient_pred = np.asarray(patient_pred, dtype=np.int32)
    patient_scores = np.asarray(patient_scores, dtype=np.float32)

    patient_acc = 100.0 * float(np.mean(patient_true == patient_pred)) if patient_true.size > 0 else 0.0

    return patient_acc, patient_true, patient_pred, patient_scores


def train():
    set_seed(42)




    config = {
        "use_2_5d": True,
        "patient_threshold": 0.5,
        "patient_top_k": 10,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_path = "data/splits/patient_split.json"




    train_loader, val_loader, test_loader = create_dataloaders(
        split_path,
        batch_size=8,
        use_2_5d=config["use_2_5d"]
    )




    x, _, *_ = next(iter(train_loader))
    print("Baseline check - Input shape:", x.shape)




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




    in_channels = 3 if config["use_2_5d"] else 1
    model = BrainMRICNN(num_classes=2, in_channels=in_channels).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=1
    )

    num_epochs = 60
    best_val_patient_acc = 0.0

    os.makedirs("checkpoints", exist_ok=True)




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

        val_loss, val_acc, val_true, _, val_probs, val_patient_ids = evaluate(
            model, val_loader, criterion, device
        )

        val_patient_acc, _, _, _ = evaluate_patient_level(
            val_true,
            val_probs,
            val_patient_ids,
            threshold=config["patient_threshold"],
            top_k=config["patient_top_k"],
        )

        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
        print(f"Val   Patient Acc: {val_patient_acc:.2f}%")
        print("-" * 50)

        if val_patient_acc > best_val_patient_acc:
            best_val_patient_acc = val_patient_acc
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print("✔ Best model saved.")




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

    print("\nROC-AUC Analysis (Slice-Level):")
    fpr, tpr, _ = roc_curve(test_true, test_probs)
    roc_auc = auc(fpr, tpr)
    print(f"ROC-AUC: {roc_auc:.4f}")

    patient_acc, patient_true, patient_pred, patient_scores = evaluate_patient_level(
        test_true,
        test_probs,
        test_patient_ids,
        threshold=config["patient_threshold"],
        top_k=config["patient_top_k"],
    )

    print("\n===== PATIENT-LEVEL EVALUATION (Top-k Aggregation) =====")
    print(f"Patient-level Accuracy: {patient_acc:.2f}%")

    print("\nPatient-level Confusion Matrix:")
    print(confusion_matrix(patient_true, patient_pred))

    print("\nPatient-level Classification Report:")
    print(classification_report(patient_true, patient_pred, zero_division=0))

    unique_patient_classes = np.unique(patient_true)
    if unique_patient_classes.size > 1:
        pfpr, ptpr, _ = roc_curve(patient_true, patient_scores)
        patient_auc = auc(pfpr, ptpr)
        print(f"\nPatient-level ROC-AUC: {patient_auc:.4f}")
    else:
        print("\nPatient-level ROC-AUC: N/A (only one class present in patient-level ground truth)")


if __name__ == "__main__":
    train()