from collections import defaultdict
import numpy as np


def topk_patient_prediction(records, probs, k=5):
    
    patient_dict = defaultdict(list)

    for record, prob in zip(records, probs):
        patient_id = record["patient_id"]
        patient_dict[patient_id].append(prob)

    patient_predictions = {}

    for patient_id, patient_probs in patient_dict.items():

        patient_probs = np.array(patient_probs)

        class1_probs = patient_probs[:, 1]

        topk_indices = np.argsort(class1_probs)[-k:]
        topk_probs = class1_probs[topk_indices]

        prediction = 1 if np.mean(topk_probs) > 0.5 else 0

        patient_predictions[patient_id] = prediction

    return patient_predictions

def get_patient_labels(records):

    patient_labels = {}

    for record in records:
        patient_id = record["patient_id"]
        label = record["label"]

        patient_labels[patient_id] = label  

    return patient_labels