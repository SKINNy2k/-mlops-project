
import json
import joblib

from mlflow_config import BASE_DIR


MODEL_PATH = BASE_DIR / "models" / "iris_model.joblib"
PROMOTION_PATH = BASE_DIR / "promotion_result.json"


def load_model():
    """Load the model only if the promotion result is approved."""

    if not PROMOTION_PATH.exists():
        raise RuntimeError("Promotion result not found")

    with PROMOTION_PATH.open("r", encoding="utf-8") as file:
        result = json.load(file)

    if result.get("status") != "APPROVED":
        raise RuntimeError("Model has not been approved")

    if not MODEL_PATH.exists():
        raise RuntimeError("Model file not found")

    return joblib.load(MODEL_PATH)


CLASS_NAMES = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}
