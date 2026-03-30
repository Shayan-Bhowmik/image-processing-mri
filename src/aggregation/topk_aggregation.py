from collections import defaultdict
import numpy as np


def aggregate_patient_tumor_score(
    tumor_probs,
    top_k=20,
    method="median"
):
    if len(tumor_probs) == 0:
        raise ValueError("tumor_probs cannot be empty")

    probs = np.asarray(tumor_probs, dtype=np.float32)
    k = max(1, min(int(top_k), len(probs)))
    topk = np.sort(probs)[-k:]

    if method == "mean":
        return float(np.mean(topk))
    if method == "median":
        return float(np.median(topk))

    raise ValueError(f"Unsupported aggregation method: {method}")


def robust_patient_prediction_from_tumor_probs(
    tumor_probs,
    threshold=0.70,
    top_k=20,
    method="median",
    min_suspicious_slices=8,
    suspicious_prob_threshold=0.90,
    min_suspicious_fraction=0.30,
):
    topk_score = aggregate_patient_tumor_score(
        tumor_probs=tumor_probs,
        top_k=top_k,
        method=method,
    )

    probs = np.asarray(tumor_probs, dtype=np.float32)
    suspicious_count = int(np.sum(probs >= float(suspicious_prob_threshold)))
    suspicious_fraction = float(suspicious_count / max(1, len(probs)))
    if min_suspicious_fraction <= 0:
        fraction_strength = 1.0
    else:
        fraction_strength = min(1.0, suspicious_fraction / float(min_suspicious_fraction))

    risk_score = float(0.5 * topk_score + 0.5 * fraction_strength)

    is_tumor = (
        (topk_score >= float(threshold))
        and (suspicious_count >= int(min_suspicious_slices))
        and (suspicious_fraction >= float(min_suspicious_fraction))
    )
    prediction = 1 if is_tumor else 0
    confidence = risk_score if is_tumor else max(1.0 - suspicious_fraction, 1.0 - topk_score)

    return {
        "prediction": prediction,
        "score": float(topk_score),
        "risk_score": risk_score,
        "confidence": float(confidence),
        "suspicious_slices": suspicious_count,
        "suspicious_fraction": suspicious_fraction,
        "threshold": float(threshold),
        "top_k": int(top_k),
        "method": method,
    }


def topk_patient_prediction(
    records,
    probs,
    k=20,
    threshold=0.70,
    method="median",
    min_suspicious_slices=8,
    suspicious_prob_threshold=0.90,
    min_suspicious_fraction=0.30,
):
    
    patient_dict = defaultdict(list)

    for record, prob in zip(records, probs):
        patient_id = record["patient_id"]
        patient_dict[patient_id].append(prob)

    patient_predictions = {}

    for patient_id, patient_probs in patient_dict.items():

        patient_probs = np.array(patient_probs)

        class1_probs = patient_probs[:, 1]

        decision = robust_patient_prediction_from_tumor_probs(
            tumor_probs=class1_probs,
            threshold=threshold,
            top_k=max(1, int(k)),
            method=method,
            min_suspicious_slices=min_suspicious_slices,
            suspicious_prob_threshold=suspicious_prob_threshold,
            min_suspicious_fraction=min_suspicious_fraction,
        )
        prediction = decision["prediction"]

        patient_predictions[patient_id] = prediction

    return patient_predictions

def get_patient_labels(records):

    patient_labels = {}

    for record in records:
        patient_id = record["patient_id"]
        label = record["label"]

        patient_labels[patient_id] = label  

    return patient_labels