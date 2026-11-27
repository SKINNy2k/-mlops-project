"""Log the already-trained model and its evaluation to MLflow (no retraining)."""
import json
import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow_config import BASE_DIR, configure_mlflow
from promotion_gate import promote_model

def track_existing_model():
    configure_mlflow()
    model_path = BASE_DIR / "models" / "iris_model.joblib"
    if not model_path.is_file(): raise FileNotFoundError(f"Model not found: {model_path}")
    model = joblib.load(model_path)
    iris = load_iris()
    _, X_test, _, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target)
    predictions = model.predict(X_test)
    metrics = {"accuracy": float(accuracy_score(y_test, predictions)),
               "precision": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
               "recall": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
               "f1_score": float(f1_score(y_test, predictions, average="macro", zero_division=0))}
    with mlflow.start_run(run_name="Existing_Iris_Model") as run:
        mlflow.log_params({"model_type": "LogisticRegression", "scaler": "StandardScaler", "test_size": 0.2, "random_state": 42})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="iris_model")
        with (BASE_DIR / "metrics.json").open("w", encoding="utf-8") as f: json.dump(metrics, f, indent=4)
        mlflow.log_artifact(str(BASE_DIR / "metrics.json"))
        run_id = run.info.run_id
    decision = promote_model(metrics, run_id)
    print(f"MLflow run: {run_id}")
    print(f"Metrics: {metrics}")
    print(f"Promotion: {decision['status']}")
    return run_id, metrics, decision

if __name__ == "__main__":
    track_existing_model()
