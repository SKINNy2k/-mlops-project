"""Metric-based model promotion gate with auditable MLflow decision tags."""
import json
from mlflow_config import BASE_DIR, configure_mlflow

MIN_ACCURACY = 0.90
MIN_F1_SCORE = 0.90
REQUIRED_METRICS = ("accuracy", "f1_score")

def check_promotion(metrics, baseline_f1=None):
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a dictionary")
    missing = [key for key in REQUIRED_METRICS if key not in metrics]
    if missing:
        raise ValueError(f"Missing required metrics: {', '.join(missing)}")
    accuracy, f1 = metrics["accuracy"], metrics["f1_score"]
    for name, value in (("accuracy", accuracy), ("f1_score", f1)):
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be a number between 0 and 1")
    if baseline_f1 is not None and (not isinstance(baseline_f1, (int, float)) or not 0 <= baseline_f1 <= 1):
        raise ValueError("baseline_f1 must be between 0 and 1")
    reasons = []
    if accuracy < MIN_ACCURACY: reasons.append("Accuracy below minimum threshold (0.90)")
    if f1 < MIN_F1_SCORE: reasons.append("F1-score below minimum threshold (0.90)")
    if baseline_f1 is not None and f1 <= baseline_f1: reasons.append("Model did not improve over previous model")
    approved = not reasons
    return {"approved": approved, "status": "APPROVED" if approved else "REJECTED",
            "reasons": reasons, "accuracy": float(accuracy), "f1_score": float(f1)}

def promote_model(metrics, run_id, baseline_f1=None):
    if not run_id or not isinstance(run_id, str):
        raise ValueError("A valid MLflow run_id is required")
    configure_mlflow()
    result = check_promotion(metrics, baseline_f1)
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    client.set_tag(run_id, "promotion_status", result["status"])
    client.set_tag(run_id, "promotion_reason", "; ".join(result["reasons"]) if result["reasons"] else "All promotion conditions passed")
    with (BASE_DIR / "promotion_result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    return result
