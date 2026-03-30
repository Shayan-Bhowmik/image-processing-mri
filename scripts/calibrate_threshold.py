import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import (
    aggregate_patient_score,
    load_trained_model,
    predict_slices,
    preprocess_uploaded_nifti,
)


@dataclass
class CaseResult:
    case_id: str
    label: int
    score: float
    num_slices: int


def find_brats_flair_files(brats_root: Path) -> list[tuple[str, int, Path]]:
    entries: list[tuple[str, int, Path]] = []
    for patient_dir in sorted(brats_root.iterdir()):
        if not patient_dir.is_dir():
            continue

        flair_file = None
        for file_path in patient_dir.iterdir():
            if file_path.is_file() and "flair" in file_path.name.lower() and file_path.suffix.lower() == ".nii":
                flair_file = file_path
                break

        if flair_file is not None:
            entries.append((patient_dir.name, 1, flair_file))

    return entries


def find_oasis_files(oasis_root: Path) -> list[tuple[str, int, Path]]:
    entries: list[tuple[str, int, Path]] = []
    for file_path in sorted(oasis_root.glob("*.nii")):
        entries.append((file_path.name, 0, file_path))
    return entries


def evaluate_case_scores(
    model,
    device,
    all_cases: list[tuple[str, int, Path]],
    top_k: int,
) -> tuple[list[CaseResult], list[tuple[str, str]]]:
    results: list[CaseResult] = []
    failures: list[tuple[str, str]] = []

    for case_id, label, file_path in all_cases:
        try:
            payload = file_path.read_bytes()
            prep = preprocess_uploaded_nifti(payload, file_path.name)
            _, probs = predict_slices(model, prep["input_batch"], device)
            score = float(aggregate_patient_score(probs, top_k=top_k))
            results.append(CaseResult(case_id=case_id, label=label, score=score, num_slices=int(len(probs))))
        except Exception as exc:  # noqa: BLE001
            failures.append((str(file_path), str(exc)))

    return results, failures


def confusion_from_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, int]:
    preds = (scores >= threshold).astype(np.int32)

    tp = int(np.sum((labels == 1) & (preds == 1)))
    fn = int(np.sum((labels == 1) & (preds == 0)))
    tn = int(np.sum((labels == 0) & (preds == 0)))
    fp = int(np.sum((labels == 0) & (preds == 1)))

    return {"tp": tp, "fn": fn, "tn": tn, "fp": fp}


def metrics_from_confusion(conf: dict[str, int]) -> dict[str, float]:
    tp, fn, tn, fp = conf["tp"], conf["fn"], conf["tn"], conf["fp"]

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_accuracy = 0.5 * (sensitivity + specificity)
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "accuracy": float(accuracy),
    }


def build_threshold_grid(scores: np.ndarray) -> np.ndarray:
    unique_scores = np.unique(np.round(scores, 6))
    dense = np.linspace(0.0, 1.0, 201)
    grid = np.unique(np.concatenate([unique_scores, dense, np.array([0.5])]))
    return np.clip(grid, 0.0, 1.0)


def pick_best_threshold(labels: np.ndarray, scores: np.ndarray, threshold_grid: np.ndarray) -> dict[str, object]:
    evaluated_rows: list[dict[str, object]] = []

    for threshold in threshold_grid:
        conf = confusion_from_threshold(labels, scores, float(threshold))
        metrics = metrics_from_confusion(conf)
        row = {
            "threshold": float(threshold),
            **conf,
            **metrics,
        }
        evaluated_rows.append(row)

    best = max(
        evaluated_rows,
        key=lambda row: (
            row["balanced_accuracy"],
            row["specificity"],
            row["sensitivity"],
            -abs(row["threshold"] - 0.5),
        ),
    )

    return {"best": best, "all": evaluated_rows}


def to_float(value: float) -> float:
    return float(np.round(value, 6))


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate patient-level threshold on combined BraTS + OASIS.")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth", help="Path to model checkpoint")
    parser.add_argument(
        "--brats-root",
        default="data/raw/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData",
        help="BraTS root directory",
    )
    parser.add_argument(
        "--oasis-root",
        default="data/raw/oasis/OASIS_Clean_Data/OASIS_Clean_Data",
        help="OASIS root directory",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-k slices used for patient score aggregation")
    parser.add_argument(
        "--out-dir",
        default="outputs/calibration",
        help="Output folder for calibration report and threshold file",
    )

    args = parser.parse_args()

    brats_root = Path(args.brats_root)
    oasis_root = Path(args.oasis_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    brats_cases = find_brats_flair_files(brats_root)
    oasis_cases = find_oasis_files(oasis_root)
    all_cases = brats_cases + oasis_cases

    if not all_cases:
        raise RuntimeError("No cases were found in BraTS/OASIS roots.")

    print(f"BraTS cases: {len(brats_cases)}")
    print(f"OASIS cases: {len(oasis_cases)}")
    print(f"Total cases: {len(all_cases)}")

    model, device = load_trained_model(checkpoint_path=args.checkpoint)
    print(f"Model loaded on: {device}")

    results, failures = evaluate_case_scores(model, device, all_cases, top_k=max(1, int(args.top_k)))

    if not results:
        raise RuntimeError("No valid cases processed. Check data paths and preprocessing.")

    labels = np.array([r.label for r in results], dtype=np.int32)
    scores = np.array([r.score for r in results], dtype=np.float32)

    thresholds = build_threshold_grid(scores)
    picked = pick_best_threshold(labels, scores, thresholds)
    best = picked["best"]

    baseline_conf = confusion_from_threshold(labels, scores, 0.5)
    baseline_metrics = metrics_from_confusion(baseline_conf)

    calibration_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": args.checkpoint,
        "top_k": int(args.top_k),
        "dataset_counts": {
            "brats": len(brats_cases),
            "oasis": len(oasis_cases),
            "processed": len(results),
            "failed": len(failures),
        },
        "score_stats": {
            "min": to_float(float(scores.min())),
            "max": to_float(float(scores.max())),
            "mean": to_float(float(scores.mean())),
            "std": to_float(float(scores.std())),
        },
        "baseline_threshold_0_5": {
            **baseline_conf,
            **{k: to_float(v) for k, v in baseline_metrics.items()},
        },
        "recommended": {
            "threshold": to_float(best["threshold"]),
            "tp": int(best["tp"]),
            "fn": int(best["fn"]),
            "tn": int(best["tn"]),
            "fp": int(best["fp"]),
            "sensitivity": to_float(best["sensitivity"]),
            "specificity": to_float(best["specificity"]),
            "balanced_accuracy": to_float(best["balanced_accuracy"]),
            "accuracy": to_float(best["accuracy"]),
        },
        "threshold_table": [
            {
                "threshold": to_float(row["threshold"]),
                "sensitivity": to_float(row["sensitivity"]),
                "specificity": to_float(row["specificity"]),
                "balanced_accuracy": to_float(row["balanced_accuracy"]),
                "accuracy": to_float(row["accuracy"]),
                "tp": int(row["tp"]),
                "fn": int(row["fn"]),
                "tn": int(row["tn"]),
                "fp": int(row["fp"]),
            }
            for row in picked["all"]
        ],
        "failures": [{"case": case, "error": err} for case, err in failures],
    }

    recommended_payload = {
        "created_at": calibration_payload["created_at"],
        "checkpoint": args.checkpoint,
        "recommended_threshold": to_float(best["threshold"]),
        "top_k": int(args.top_k),
        "metrics": calibration_payload["recommended"],
    }

    report_path = out_dir / "threshold_report.json"
    recommendation_path = out_dir / "recommended_threshold.json"

    report_path.write_text(json.dumps(calibration_payload, indent=2), encoding="utf-8")
    recommendation_path.write_text(json.dumps(recommended_payload, indent=2), encoding="utf-8")

    print("\n=== Combined BraTS + OASIS Threshold Tuning ===")
    print(f"Processed cases: {len(results)} (failed: {len(failures)})")
    print("\nBaseline @ threshold 0.50")
    print(
        f"Sensitivity={baseline_metrics['sensitivity']:.4f} | "
        f"Specificity={baseline_metrics['specificity']:.4f} | "
        f"Balanced Acc={baseline_metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"Confusion: TP={baseline_conf['tp']} FN={baseline_conf['fn']} "
        f"TN={baseline_conf['tn']} FP={baseline_conf['fp']}"
    )

    print("\nRecommended threshold")
    print(
        f"Threshold={best['threshold']:.4f} | Sensitivity={best['sensitivity']:.4f} | "
        f"Specificity={best['specificity']:.4f} | Balanced Acc={best['balanced_accuracy']:.4f}"
    )
    print(
        f"Confusion: TP={best['tp']} FN={best['fn']} TN={best['tn']} FP={best['fp']}"
    )

    print(f"\nSaved report: {report_path}")
    print(f"Saved recommended threshold: {recommendation_path}")


if __name__ == "__main__":
    main()
