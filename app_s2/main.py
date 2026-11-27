from fastapi import FastAPI, HTTPException

from app_s2.schemas import IrisFeatures, PredictionResponse
from app_s2.model_loader import load_model, CLASS_NAMES

app = FastAPI(
    title="Iris Classification API",
    description="MLflow-tracked Iris classification service",
    version="1.0.0"
)

# Load the approved model when the application starts
try:
    model = load_model()
except Exception as exc:
    raise RuntimeError(f"Could not load model: {exc}") from exc


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
    try:
        input_data = [[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]]

        prediction = int(model.predict(input_data)[0])

        return PredictionResponse(
            prediction=prediction,
            class_name=CLASS_NAMES[prediction]
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}"
        )