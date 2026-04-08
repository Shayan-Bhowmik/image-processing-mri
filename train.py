import argparse
import time
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

from models.model import BrainMRICNN, MultiModalBrainMRI, FlexibleMultiModalBrainMRI
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


def train(
    split_path="data/splits/patient_split.json",
    canonical_shape=(192, 192, 160),
    fixed_slice_count=96,
    exclude_brats2021=False,
):
    set_seed(42)




    config = {
        "use_multimodal": True,
        "patient_threshold": 0.5,
        "patient_top_k": 10,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = create_dataloaders(
        split_path,
        batch_size=4,
        use_2_5d=config.get("use_multimodal", True),
        canonical_shape=canonical_shape,
        fixed_slice_count=fixed_slice_count,
        exclude_brats2021=exclude_brats2021,
    )




    x, _, *_ = next(iter(train_loader))
    print("Baseline check - Input shape:", x.shape)

    num_modalities = x.shape[1]
    print(f"Detected {num_modalities} modalities in dataset\n")




    # Build class labels directly from indexed metadata to avoid a full lazy-loading pass.
    all_labels = [label for _, _, label in train_loader.dataset.index_map]

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




    modality_dropout_rate = 0.1
    model = FlexibleMultiModalBrainMRI(
        num_classes=2,
        num_modalities=num_modalities,
        modality_dropout_rate=modality_dropout_rate,
    ).to(device)

    print(f"\n{'='*70}")
    print("FLEXIBLE MULTI-MODAL MRI MODEL WITH WEIGHTED FUSION")
    print(f"{'='*70}")
    print(f"Number of modalities (auto-detected): {num_modalities}")
    print(f"Modality dropout rate (for robustness): {modality_dropout_rate}")
    print("Architecture: Adaptive Conv + Weighted Fusion + 3-layer CNN")
    print("Model adapts to ANY number of modalities automatically!")
    print(f"{'='*70}\n")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=1
    )

    num_epochs = 20
    best_val_patient_acc = 0.0

    os.makedirs("checkpoints", exist_ok=True)

    print("\n" + "="*70)
    print("STARTING TRAINING LOOP")
    print("="*70 + "\n")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"Epoch {epoch + 1}/{num_epochs} - Starting batch processing...")

        for batch_idx, (images, labels, _) in enumerate(train_loader):
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}", end='\r')
            
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, apply_modality_dropout=True)
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
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_seconds = time.time() - epoch_start_time

        if (epoch + 1) % 5 == 0 or epoch == 0:
            modality_weights = model.get_modality_weights(
                modality_names=[f"M{i}" for i in range(num_modalities)]
            )
            print(f"\nLearned Modality Weights (epoch {epoch+1}):")
            for mod, weight in modality_weights.items():
                bar_length = int(weight * 30)
                bar = "█" * bar_length + "░" * (30 - bar_length)
                print(f"  {mod:8s}: {weight:.4f} |{bar}|")

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
        print(f"Val   Patient Acc: {val_patient_acc:.2f}%")
        print(f"Epoch Time: {epoch_seconds:.1f}s | LR: {current_lr:.6f}")
        print("-" * 70)

        if val_patient_acc > best_val_patient_acc:
            best_val_patient_acc = val_patient_acc
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print("✔ Best model saved.")




    print("\n" + "="*70)
    print("FINAL LEARNED MODALITY WEIGHTS")
    print("="*70)
    modality_weights = model.get_modality_weights(
        modality_names=[f"Modality_{i}" for i in range(num_modalities)]
    )
    for mod, weight in modality_weights.items():
        bar_length = int(weight * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"{mod:15s}: {weight:.4f} |{bar}|")
    print("="*70 + "\n")

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
    parser = argparse.ArgumentParser(description="Train MRI classifier")
    parser.add_argument(
        "--split-path",
        default="data/splits/patient_split.json",
        help="Path to split JSON file",
    )
    parser.add_argument(
        "--canonical-shape",
        type=int,
        nargs=3,
        default=(192, 192, 160),
        metavar=("H", "W", "D"),
        help="Canonical 3D shape used before slice extraction",
    )
    parser.add_argument(
        "--fixed-slice-count",
        type=int,
        default=96,
        help="Maximum valid slices sampled per volume",
    )
    parser.add_argument(
        "--exclude-brats2021",
        action="store_true",
        help="Exclude BRATS 2021 dataset (use only BRATS 2020 + OASIS for faster training)",
    )

    args = parser.parse_args()
    train(
        split_path=args.split_path,
        canonical_shape=tuple(args.canonical_shape),
        fixed_slice_count=int(args.fixed_slice_count),
        exclude_brats2021=args.exclude_brats2021,
    )