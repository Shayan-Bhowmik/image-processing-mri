from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def _safe_auc(labels, probs):
    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, probs))


def fit_temperature(logits_tensor, labels_tensor, max_iter=50):
    # Learn a single temperature on validation logits for probability calibration.
    log_temp = torch.nn.Parameter(torch.zeros(1, device=logits_tensor.device))
    optimizer = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temp)
        loss = torch.nn.functional.cross_entropy(logits_tensor / temperature, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temp).item())


def find_optimal_threshold(labels, probs, min_threshold=0.60):
    # Use Youden's J statistic on validation set, then enforce a stricter lower bound.
    if len(set(labels)) < 2:
        return min_threshold

    fpr, tpr, thresholds = roc_curve(labels, probs)
    scores = tpr - fpr
    best_idx = int(np.argmax(scores))
    raw_thr = float(thresholds[best_idx])
    return float(max(raw_thr, min_threshold))


def _patient_level_aggregate(patient_probs, patient_labels):
    patient_true_labels = []
    patient_tumor_probs = []

    for pid in patient_probs:
        mean_prob = float(np.mean(patient_probs[pid]))
        labels_for_patient = patient_labels[pid]
        majority_label = 1 if sum(labels_for_patient) >= (len(labels_for_patient) / 2) else 0
        patient_tumor_probs.append(mean_prob)
        patient_true_labels.append(int(majority_label))

    return patient_true_labels, patient_tumor_probs


def evaluate_dataset(
    model,
    dataloader,
    criterion,
    device,
    threshold=None,
    min_threshold=0.50,
    temperature=None,
    fit_calibration=False,
    split_name="Validation",
):
    model.eval()

    total_loss = 0.0
    total = 0

    sample_labels = []
    sample_probs = []
    raw_logits = []

    patient_probs = defaultdict(list)
    patient_labels = defaultdict(list)

    with torch.no_grad():
        running_sample_id = 0

        for batch in dataloader:
            if len(batch) == 3:
                images, labels, patient_ids = batch
            else:
                images, labels = batch
                patient_ids = [f"sample_{running_sample_id + i}" for i in range(labels.size(0))]

            running_sample_id += labels.size(0)
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += float(loss.item())
            total += labels.size(0)

            raw_logits.append(outputs.detach().cpu())
            labels_np = labels.cpu().numpy()

            sample_labels.extend(labels_np.tolist())

            probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
            sample_probs.extend(probs.tolist())

            for pid, label_value, prob_value in zip(patient_ids, labels_np, probs):
                patient_probs[str(pid)].append(float(prob_value))
                patient_labels[str(pid)].append(int(label_value))

    if total == 0:
        raise ValueError(f"{split_name} dataloader is empty.")

    logits_tensor = torch.cat(raw_logits, dim=0)
    labels_tensor = torch.tensor(sample_labels, dtype=torch.long)

    used_temperature = 1.0 if temperature is None else float(temperature)
    if fit_calibration:
        used_temperature = fit_temperature(logits_tensor, labels_tensor)

    calibrated_probs = torch.softmax(logits_tensor / used_temperature, dim=1)[:, 1].numpy()

    # Rebuild patient probabilities with calibrated values.
    patient_probs_cal = defaultdict(list)
    idx = 0
    for pid in patient_probs:
        count = len(patient_probs[pid])
        patient_probs_cal[pid] = calibrated_probs[idx: idx + count].tolist()
        idx += count

    patient_true_labels, patient_tumor_probs = _patient_level_aggregate(patient_probs_cal, patient_labels)

    used_threshold = threshold
    if used_threshold is None:
        used_threshold = find_optimal_threshold(
            patient_true_labels,
            patient_tumor_probs,
            min_threshold=min_threshold,
        )

    patient_preds = [1 if p >= used_threshold else 0 for p in patient_tumor_probs]

    tn, fp, fn, tp = confusion_matrix(
        patient_true_labels,
        patient_preds,
        labels=[0, 1],
    ).ravel()

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    accuracy = float((tp + tn) / max(1, (tp + tn + fp + fn)))
    balanced_accuracy = float((sensitivity + specificity) / 2.0)

    sample_auc = _safe_auc(sample_labels, calibrated_probs.tolist())
    patient_auc = _safe_auc(patient_true_labels, patient_tumor_probs)

    metrics = {
        "split": split_name,
        "loss": float(total_loss / len(dataloader)),
        "sample_auc": sample_auc,
        "patient_auc": patient_auc,
        "threshold": float(used_threshold),
        "temperature": float(used_temperature),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "n_patients": int(len(patient_true_labels)),
    }

    return metrics


def print_metrics(metrics):
    cm = metrics["confusion_matrix"]
    print(f"[{metrics['split']}] Loss: {metrics['loss']:.4f}")
    print(
        f"[{metrics['split']}] Accuracy: {metrics['accuracy']:.4f} | "
        f"Balanced Acc: {metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"[{metrics['split']}] Sensitivity: {metrics['sensitivity']:.4f} | "
        f"Specificity: {metrics['specificity']:.4f}"
    )
    print(
        f"[{metrics['split']}] ROC AUC (sample/patient): "
        f"{metrics['sample_auc']} / {metrics['patient_auc']}"
    )
    print(
        f"[{metrics['split']}] Threshold: {metrics['threshold']:.4f} | "
        f"Temperature: {metrics['temperature']:.4f}"
    )
    print(
        f"[{metrics['split']}] Confusion Matrix [tn fp; fn tp]: "
        f"[{cm['tn']} {cm['fp']}; {cm['fn']} {cm['tp']}]"
    )