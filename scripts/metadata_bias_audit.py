import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


def resolve_case_path(entry, brats_root: Path, oasis_root: Path) -> Path | None:
    case_id = str(entry["id"])
    label = int(entry["label"])

    if label == 1:
        patient_dir = brats_root / case_id
        if not patient_dir.exists():
            return None

        for file_name in sorted(os.listdir(patient_dir)):
            file_lower = file_name.lower()
            if "flair" in file_lower and file_lower.endswith((".nii", ".nii.gz")):
                return patient_dir / file_name
        return None

    file_path = oasis_root / case_id
    return file_path if file_path.exists() else None


def extract_header_features(file_path: Path) -> np.ndarray:
    nii = nib.load(str(file_path))
    shape = nii.shape
    shape3 = tuple(int(v) for v in shape[:3])
    zoom3 = tuple(float(v) for v in nii.header.get_zooms()[:3])
    return np.array([*shape3, *zoom3], dtype=np.float32)


def build_matrix(entries, brats_root: Path, oasis_root: Path):
    features = []
    labels = []

    for entry in entries:
        file_path = resolve_case_path(entry, brats_root, oasis_root)
        if file_path is None:
            continue

        features.append(extract_header_features(file_path))
        labels.append(int(entry["label"]))

    if not features:
        raise RuntimeError("No valid entries resolved from split file.")

    return np.vstack(features), np.array(labels, dtype=np.int32)


def depth_rule_predict(x: np.ndarray, threshold: float) -> np.ndarray:
    depth = x[:, 2]
    return (depth < threshold).astype(np.int32)


def fit_best_depth_threshold(x_train: np.ndarray, y_train: np.ndarray) -> float:
    depth_vals = np.unique(x_train[:, 2])
    best_acc = -1.0
    best_t = float(depth_vals[0])

    for threshold in depth_vals:
        pred = depth_rule_predict(x_train, float(threshold))
        acc = float(np.mean(pred == y_train))
        if acc > best_acc:
            best_acc = acc
            best_t = float(threshold)

    return best_t


def eval_and_print(name: str, y_true: np.ndarray, y_pred: np.ndarray):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f"{name} accuracy: {acc:.4f}")
    print(f"{name} confusion matrix: {cm.tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit dataset shortcut bias using metadata-only baselines.")
    parser.add_argument("--split-json", default="data/splits/patient_split.json")
    parser.add_argument(
        "--brats-root",
        default="data/raw/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData",
    )
    parser.add_argument(
        "--oasis-root",
        default="data/raw/oasis/OASIS_Clean_Data/OASIS_Clean_Data",
    )
    args = parser.parse_args()

    split = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
    brats_root = Path(args.brats_root)
    oasis_root = Path(args.oasis_root)

    x_train, y_train = build_matrix(split["train"], brats_root, oasis_root)
    x_val, y_val = build_matrix(split["val"], brats_root, oasis_root)
    x_test, y_test = build_matrix(split["test"], brats_root, oasis_root)

    print("=== Metadata Bias Audit ===")
    print("Feature vector: [shape_x, shape_y, shape_z, zoom_x, zoom_y, zoom_z]")

    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_train, y_train)

    print("\nLogistic regression baseline (metadata only):")
    eval_and_print("Train", y_train, clf.predict(x_train))
    eval_and_print("Val", y_val, clf.predict(x_val))
    eval_and_print("Test", y_test, clf.predict(x_test))

    threshold = fit_best_depth_threshold(x_train, y_train)
    print("\nSingle-feature depth rule baseline:")
    print(f"Rule: predict tumor if depth < {threshold:g}")
    eval_and_print("Train", y_train, depth_rule_predict(x_train, threshold))
    eval_and_print("Val", y_val, depth_rule_predict(x_val, threshold))
    eval_and_print("Test", y_test, depth_rule_predict(x_test, threshold))


if __name__ == "__main__":
    main()
