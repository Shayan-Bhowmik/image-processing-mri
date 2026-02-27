import json
from collections import Counter

with open("data/splits/patient_split.json") as f:
    split = json.load(f)

for subset in ["train", "val", "test"]:
    labels = [p["label"] for p in split[subset]]
    print(subset, Counter(labels))