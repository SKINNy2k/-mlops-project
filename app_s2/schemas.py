
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


class IrisFeatures(BaseModel):
    """Input features required for Iris prediction."""

    model_config = ConfigDict(extra="forbid")

    sepal_length: FiniteFloat = Field(gt=0)
    sepal_width: FiniteFloat = Field(gt=0)
    petal_length: FiniteFloat = Field(gt=0)
    petal_width: FiniteFloat = Field(gt=0)


class PredictionResponse(BaseModel):
    """Response returned by the prediction endpoint."""

    prediction: int
    class_name: str
