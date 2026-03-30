import argparse
import gzip
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.aggregation.topk_aggregation import robust_patient_prediction_from_tumor_probs
from src.dataset.input_transforms import build_eval_transform
from src.dataset.mri_dataset import MRISliceDataset
from src.dataset.split_utils import split_dataset_by_patient
from src.evaluation.predictor import Predictor


def load_dataset(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def get_latest_checkpoint(checkpoint_dir="outputs/checkpoints"):
    checkpoint_dir = Path(checkpoint_dir)
    candidates = list(checkpoint_dir.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)


def build_patient_view(records, probabilities):
    patient_probs = defaultdict(list)
    patient_labels = {}

    for record, prob in zip(records, probabilities):
        patient_id = record["patient_id"]
        patient_probs[patient_id].append(float(prob[1]))
        patient_labels[patient_id] = int(record["label"])

    patient_ids = sorted(patient_probs.keys())
    return patient_ids, patient_probs, patient_labels


def evaluate_params(patient_ids, patient_probs, patient_labels, params):
    y_true = []
    y_pred = []

    for pid in patient_ids:
        decision = robust_patient_prediction_from_tumor_probs(
            tumor_probs=patient_probs[pid],
            threshold=params["threshold"],
            top_k=params["top_k"],
            method=params["method"],
            min_suspicious_slices=params["min_suspicious_slices"],
            suspicious_prob_threshold=params["suspicious_prob_threshold"],
            min_suspicious_fraction=params["min_suspicious_fraction"],
        )
        y_true.append(patient_labels[pid])
        y_pred.append(int(decision["prediction"]))

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    balanced_accuracy = 0.5 * (sensitivity + specificity)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
    }


def grid_search(patient_ids, patient_probs, patient_labels):
    thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    top_ks = [10, 15, 20, 25, 30]
    methods = ["median", "mean"]
    min_suspicious_slices_values = [4, 6, 8, 10, 12, 15]
    suspicious_prob_thresholds = [0.85, 0.90, 0.95]
    min_suspicious_fractions = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    results = []

    for threshold in thresholds:
        for top_k in top_ks:
            for method in methods:
                for min_suspicious_slices in min_suspicious_slices_values:
                    for suspicious_prob_threshold in suspicious_prob_thresholds:
                        for min_suspicious_fraction in min_suspicious_fractions:
                            params = {
                                "threshold": threshold,
                                "top_k": top_k,
                                "method": method,
                                "min_suspicious_slices": min_suspicious_slices,
                                "suspicious_prob_threshold": suspicious_prob_threshold,
                                "min_suspicious_fraction": min_suspicious_fraction,
                            }
                            metrics = evaluate_params(patient_ids, patient_probs, patient_labels, params)
                            results.append({"params": params, "metrics": metrics})

    ranked = sorted(
        results,
        key=lambda x: (
            x["metrics"]["balanced_accuracy"],
            x["metrics"]["specificity"],
            x["metrics"]["sensitivity"],
            x["params"]["threshold"],
            x["params"]["min_suspicious_fraction"],
            x["params"]["min_suspicious_slices"],
            x["params"]["top_k"],
        ),
        reverse=True,
    )

    best_with_guard = [
        r for r in ranked
        if r["metrics"]["sensitivity"] >= 0.95 and r["metrics"]["specificity"] >= 0.95
    ]

    best = best_with_guard[0] if best_with_guard else ranked[0]
    return best, ranked[:20]


def main():
    parser = argparse.ArgumentParser(description="Calibrate patient aggregation parameters")
    parser.add_argument("--dataset_path", type=str, default="data/dataset_records.pkl.gz")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output", type=str, default="outputs/calibration/aggregation_calibration.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    records = load_dataset(args.dataset_path)
    train_records, val_records = split_dataset_by_patient(records)

    print("Train distribution:", Counter([r["label"] for r in train_records]))
    print("Val distribution:", Counter([r["label"] for r in val_records]))

    val_transform = build_eval_transform(target_size=224, center_crop_size=180)
    val_dataset = MRISliceDataset(val_records, target_size=224, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    checkpoint_path = args.checkpoint or get_latest_checkpoint("outputs/checkpoints")
    print(f"Using checkpoint: {checkpoint_path}")

    predictor = Predictor.load_from_checkpoint(checkpoint_path=checkpoint_path, device=device)
    predictor.model.eval()
    outputs = predictor.collect_predictions(val_loader)

    patient_ids, patient_probs, patient_labels = build_patient_view(val_records, outputs["probabilities"])
    print(f"Patients evaluated: {len(patient_ids)}")

    best, top_results = grid_search(patient_ids, patient_probs, patient_labels)

    payload = {
        "checkpoint": checkpoint_path,
        "best": best,
        "top_results": top_results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\nBest parameters:")
    print(json.dumps(best["params"], indent=2))
    print("Best metrics:")
    print(json.dumps(best["metrics"], indent=2))
    print(f"Saved calibration report: {output_path}")


if __name__ == "__main__":
    main()
