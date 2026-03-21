from src.evaluation.metrics import compute_classification_metrics


def test_metrics():

    true_labels = [0, 1, 0, 1, 1]
    predicted_labels = [0, 1, 0, 0, 1]

    results = compute_classification_metrics(true_labels, predicted_labels)

    assert "accuracy" in results
    assert "precision" in results
    assert "recall" in results
    assert "f1_score" in results
    assert "confusion_matrix" in results

    print("Metrics test passed.")
    print(results)


if __name__ == "__main__":
    test_metrics()