# Student 1 — ML Pipeline & MLflow

## Project: Iris Flower Classification

This component trains and evaluates a three-class Iris classifier, tracks experiments with MLflow, and applies a metric-based promotion gate.

### Dataset and model
- Scikit-learn Iris dataset: 150 samples, 4 numerical features, 3 species.
- Stratified train/test split: 80% / 20%, `random_state=42`.
- Pipeline: `StandardScaler` + `LogisticRegression(max_iter=200)`.

### Files
- `train.py`: training, evaluation metrics, model/artifact logging.
- `evaluate.py`: evaluation of the saved model.
- `mlflow_config.py`: portable SQLite tracking and artifact paths.
- `track_existing_model.py`: tracks the existing model without retraining.
- `promotion_gate.py`: validates metrics and records APPROVED/REJECTED decision tags in MLflow.
- `models/iris_model.joblib`: trained model used by the downstream API.
- `tests/`: evaluation and promotion-gate unit tests.
- `mlflow.db`, `mlruns/`: experiment metadata and logged artifacts from the supplied run.

### Setup
Run these commands from this directory (`student1/`):

```bash
python -m pip install -r requirements.txt
python train.py
python evaluate.py
python track_existing_model.py
python -m pytest -q
```

`train.py` trains a new run. `track_existing_model.py` evaluates and logs the already saved model and does not retrain it.

### Promotion policy
A model is APPROVED only when Accuracy >= 0.90 and macro F1 >= 0.90. If a baseline F1 is supplied, the new model must also exceed it. The decision is saved to `promotion_result.json` and recorded as MLflow run tags. This implementation records a promotion decision; it does not register or deploy a model to a production serving environment.

### Model input order
`[sepal_length, sepal_width, petal_length, petal_width]`

```python
import joblib
model = joblib.load("models/iris_model.joblib")
prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])
print(prediction)
```
