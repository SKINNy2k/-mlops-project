"""Evaluate the saved Iris model on the fixed stratified test split."""
from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from mlflow_config import BASE_DIR

def evaluate_model(model_path=None):
    path = Path(model_path) if model_path is not None else BASE_DIR / "models" / "iris_model.joblib"
    if not path.is_file():
        raise FileNotFoundError(f"Trained model not found: {path}")
    iris = load_iris()
    _, X_test, _, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target)
    model = joblib.load(path)
    predictions = model.predict(X_test)
    results = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "recall": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
    }
    print("Evaluation Results:")
    for metric, value in results.items(): print(f"{metric}: {value:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=iris.target_names, zero_division=0))
    return results

if __name__ == "__main__":
    evaluate_model()
