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


def _is_nifti_file(file_path: Path) -> bool:
    return file_path.is_file() and file_path.name.lower().endswith((".nii", ".nii.gz"))


def _find_brats_flair_file(patient_dir: Path) -> Path | None:
    for file_path in patient_dir.iterdir():
        if _is_nifti_file(file_path) and "flair" in file_path.name.lower():
            return file_path
    return None


def find_brats_flair_files(brats_root: Path) -> list[tuple[str, int, Path]]:
    entries: list[tuple[str, int, Path]] = []
    for patient_dir in sorted(brats_root.iterdir()):
        if not patient_dir.is_dir():
            continue

        flair_file = _find_brats_flair_file(patient_dir)

        if flair_file is not None:
            entries.append((patient_dir.name, 1, flair_file))

    return entries


def find_oasis_files(oasis_root: Path) -> list[tuple[str, int, Path]]:
    entries: list[tuple[str, int, Path]] = []
    for file_path in sorted(oasis_root.iterdir()):
        if _is_nifti_file(file_path):
            entries.append((file_path.name, 0, file_path))
    return entries


def load_split_entries(split_json_path: Path, split_name: str) -> list[dict[str, object]]:
    if not split_json_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_json_path}")

    payload = json.loads(split_json_path.read_text(encoding="utf-8"))
    if split_name not in payload:
        available = ", ".join(sorted(payload.keys()))
        raise KeyError(f"Split '{split_name}' not found. Available: {available}")

    split_entries = payload[split_name]
    if not isinstance(split_entries, list):
        raise ValueError(f"Split '{split_name}' must be a list of entries.")

    return split_entries


def resolve_cases_from_split_entries(
    split_entries: list[dict[str, object]],
    brats_root: Path,
    oasis_root: Path,
) -> tuple[list[tuple[str, int, Path]], list[str]]:
    resolved: list[tuple[str, int, Path]] = []
    missing: list[str] = []

    for entry in split_entries:
        case_id = str(entry.get("id", "")).strip()
        label = int(entry.get("label", -1))

        if not case_id or label not in (0, 1):
            missing.append(f"invalid entry: {entry}")
            continue

        if label == 1:
            patient_dir = brats_root / case_id
            if not patient_dir.exists() or not patient_dir.is_dir():
                missing.append(f"missing brats patient dir: {patient_dir}")
                continue

            flair_file = _find_brats_flair_file(patient_dir)
            if flair_file is None:
                missing.append(f"missing brats flair file: {patient_dir}")
                continue

            resolved.append((case_id, label, flair_file))
        else:
            oasis_file = oasis_root / case_id
            if not _is_nifti_file(oasis_file):
                missing.append(f"missing oasis file: {oasis_file}")
                continue

            resolved.append((case_id, label, oasis_file))

    return resolved, missing


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
    parser = argparse.ArgumentParser(
        description="Calibrate patient-level threshold on a held-out split (default) or full BraTS + OASIS."
    )
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
    parser.add_argument(
        "--split-json",
        default="data/splits/patient_split.json",
        help="Patient split JSON used for held-out calibration",
    )
    parser.add_argument(
        "--split-name",
        default="val",
        choices=["train", "val", "test"],
        help="Held-out split name for threshold calibration",
    )
    parser.add_argument(
        "--use-all-cases",
        action="store_true",
        help="Use every discovered BraTS+OASIS case instead of a held-out split",
    )

    args = parser.parse_args()

    brats_root = Path(args.brats_root)
    oasis_root = Path(args.oasis_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_json_path = Path(args.split_json)

    if args.use_all_cases:
        brats_cases = find_brats_flair_files(brats_root)
        oasis_cases = find_oasis_files(oasis_root)
        all_cases = brats_cases + oasis_cases
        missing_cases: list[str] = []
        calibration_scope = "all_cases"
    else:
        split_entries = load_split_entries(split_json_path, args.split_name)
        all_cases, missing_cases = resolve_cases_from_split_entries(split_entries, brats_root, oasis_root)
        calibration_scope = f"{args.split_name}_split"

    brats_count = sum(1 for _, label, _ in all_cases if label == 1)
    oasis_count = sum(1 for _, label, _ in all_cases if label == 0)

    if not all_cases:
        raise RuntimeError("No cases were found in BraTS/OASIS roots.")

    print(f"Calibration scope: {calibration_scope}")
    print(f"BraTS cases: {brats_count}")
    print(f"OASIS cases: {oasis_count}")
    print(f"Total cases: {len(all_cases)}")
    if missing_cases:
        print(f"Missing/invalid split entries skipped: {len(missing_cases)}")

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
        "calibration_scope": calibration_scope,
        "split_json": str(split_json_path),
        "split_name": args.split_name,
        "use_all_cases": bool(args.use_all_cases),
        "dataset_counts": {
            "brats": brats_count,
            "oasis": oasis_count,
            "processed": len(results),
            "failed": len(failures),
            "missing_split_entries": len(missing_cases),
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
        "missing_split_entries": missing_cases,
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
