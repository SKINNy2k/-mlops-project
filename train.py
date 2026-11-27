"""Train and track the Iris classification pipeline."""
import json
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from mlflow_config import BASE_DIR, configure_mlflow

MODEL_PATH = BASE_DIR / "models" / "iris_model.joblib"

def train_model():
    configure_mlflow()
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
    )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=200))
    ])
    with mlflow.start_run() as run:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
            "recall": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
            "f1_score": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        }
        mlflow.log_params({"model_type": "LogisticRegression", "test_size": 0.2,
                           "random_state": 42, "max_iter": 200, "scaler": "StandardScaler"})
        mlflow.log_metrics(metrics)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        mlflow.sklearn.log_model(model, artifact_path="iris_model")
        cm = confusion_matrix(y_test, predictions)
        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
        display.plot()
        plt.title("Iris Classification Confusion Matrix")
        image_path = BASE_DIR / "confusion_matrix.png"
        plt.savefig(image_path, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(str(image_path))
        with (BASE_DIR / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        print(f"Run ID: {run.info.run_id}")
        print(f"Metrics: {metrics}")
    return model, metrics

if __name__ == "__main__":
    train_model()
