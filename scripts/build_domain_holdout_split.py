import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


def split_list(items, train_ratio=0.7, val_ratio=0.15):
    total = len(items)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)
    return items[:train_end], items[train_end:val_end], items[val_end:]


def split_train_val_only(items, train_ratio=0.7, val_ratio=0.15):
    total = len(items)
    denom = max(train_ratio + val_ratio, 1e-8)
    train_fraction = train_ratio / denom
    train_end = int(train_fraction * total)
    return items[:train_end], items[train_end:]


def get_brats_entries(brats_root: Path):
    entries = []
    for name in sorted(os.listdir(brats_root)):
        if (brats_root / name).is_dir():
            entries.append({"id": name, "label": 1})
    return entries


def get_brats_entries_multi(roots):
    merged = {}
    for root in roots:
        if not root.exists():
            continue
        for entry in get_brats_entries(root):
            merged[entry["id"]] = entry
    return sorted(merged.values(), key=lambda x: x["id"])


def group_oasis_by_subject_and_protocol(oasis_root: Path):
    subject_pattern = re.compile(r"^(OAS\d?_\d{4})_")
    protocol_pattern = re.compile(r"_MR(\d+)_", flags=re.IGNORECASE)

    grouped = defaultdict(list)
    protocols_by_subject = defaultdict(set)

    for file_name in sorted(os.listdir(oasis_root)):
        file_lower = file_name.lower()
        if not file_lower.endswith((".nii", ".nii.gz")):
            continue

        subject_match = subject_pattern.match(file_name)
        subject_id = subject_match.group(1) if subject_match else file_name

        proto_match = protocol_pattern.search(file_name)
        protocol = f"MR{proto_match.group(1)}" if proto_match else "UNKNOWN"

        grouped[subject_id].append(file_name)
        protocols_by_subject[subject_id].add(protocol.upper())

    return grouped, protocols_by_subject


def expand_oasis(subject_ids, grouped):
    entries = []
    for subject_id in subject_ids:
        for file_name in sorted(grouped[subject_id]):
            entries.append({"id": file_name, "label": 0})
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build split with OASIS protocol holdout for stronger cross-domain evaluation."
    )
    parser.add_argument(
        "--brats-root",
        default="data/raw/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData",
    )
    parser.add_argument(
        "--brats2021-root",
        default="data/raw/brats2021_extracted",
    )
    parser.add_argument(
        "--oasis-root",
        default="data/raw/oasis/OASIS_Clean_Data/OASIS_Clean_Data",
    )
    parser.add_argument("--holdout-protocol", default="MR2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--out", default="data/splits/patient_split_domain_holdout.json")
    args = parser.parse_args()

    random.seed(args.seed)

    brats_root = Path(args.brats_root)
    brats2021_root = Path(args.brats2021_root)
    oasis_root = Path(args.oasis_root)

    brats_entries = get_brats_entries_multi([brats_root, brats2021_root])
    random.shuffle(brats_entries)
    brats_train, brats_val, brats_test = split_list(
        brats_entries,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    oasis_grouped, protocols_by_subject = group_oasis_by_subject_and_protocol(oasis_root)

    holdout_subjects = [
        subject
        for subject, protocols in protocols_by_subject.items()
        if args.holdout_protocol.upper() in protocols
    ]
    non_holdout_subjects = [
        subject
        for subject, protocols in protocols_by_subject.items()
        if args.holdout_protocol.upper() not in protocols
    ]

    if not holdout_subjects:
        raise RuntimeError(f"No OASIS subjects found for holdout protocol {args.holdout_protocol}")

    random.shuffle(non_holdout_subjects)
    oasis_train_subjects, oasis_val_subjects = split_train_val_only(
        non_holdout_subjects,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    oasis_train = expand_oasis(oasis_train_subjects, oasis_grouped)
    oasis_val = expand_oasis(oasis_val_subjects, oasis_grouped)
    oasis_test = expand_oasis(holdout_subjects, oasis_grouped)

    train_entries = brats_train + oasis_train
    val_entries = brats_val + oasis_val
    test_entries = brats_test + oasis_test

    random.shuffle(train_entries)
    random.shuffle(val_entries)
    random.shuffle(test_entries)

    split = {
        "train": train_entries,
        "val": val_entries,
        "test": test_entries,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(split, indent=4), encoding="utf-8")

    print("Saved split:", out_path)
    print("Holdout protocol:", args.holdout_protocol)
    print("Split sizes:", {k: len(v) for k, v in split.items()})

    for subset in ["train", "val", "test"]:
        label_counter = Counter(int(e["label"]) for e in split[subset])
        print(subset, dict(label_counter))

    subject_pattern = re.compile(r"^(OAS\d?_\d{4})_")
    oasis_subjects = {}
    for subset in ["train", "val", "test"]:
        cur = set()
        for entry in split[subset]:
            if int(entry["label"]) != 0:
                continue
            m = subject_pattern.match(entry["id"])
            if m:
                cur.add(m.group(1))
        oasis_subjects[subset] = cur

    print(
        "OASIS subject overlap (train-val/train-test/val-test):",
        len(oasis_subjects["train"] & oasis_subjects["val"]),
        len(oasis_subjects["train"] & oasis_subjects["test"]),
        len(oasis_subjects["val"] & oasis_subjects["test"]),
    )


if __name__ == "__main__":
    main()
