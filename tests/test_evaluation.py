from evaluate import evaluate_model

def test_evaluation_metrics():
    results = evaluate_model()
    assert set(("accuracy", "precision", "recall", "f1_score")).issubset(results)
    assert all(0 <= results[name] <= 1 for name in ("accuracy", "precision", "recall", "f1_score"))

def test_model_accuracy_threshold():
    assert evaluate_model()["accuracy"] >= 0.90
