
import pytest
from fastapi.testclient import TestClient

from app_s2.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_predict_setosa():
    response = client.post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
    )

    assert response.status_code == 200
    assert response.json()["prediction"] == 0
    assert response.json()["class_name"] == "setosa"


def test_predict_versicolor():
    response = client.post(
        "/predict",
        json={
            "sepal_length": 6.0,
            "sepal_width": 2.9,
            "petal_length": 4.5,
            "petal_width": 1.5
        }
    )

    assert response.status_code == 200
    assert response.json()["prediction"] == 1
    assert response.json()["class_name"] == "versicolor"


def test_predict_virginica():
    response = client.post(
        "/predict",
        json={
            "sepal_length": 6.5,
            "sepal_width": 3.0,
            "petal_length": 5.5,
            "petal_width": 2.0
        }
    )

    assert response.status_code == 200
    assert response.json()["prediction"] == 2
    assert response.json()["class_name"] == "virginica"


def test_missing_feature():
    response = client.post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4
        }
    )

    assert response.status_code == 422


def test_extra_feature():
    response = client.post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
            "extra": 10
        }
    )

    assert response.status_code == 422


def test_negative_measurement():
    response = client.post(
        "/predict",
        json={
            "sepal_length": -5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
    )

    assert response.status_code == 422


def test_zero_measurement():
    response = client.post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 0,
            "petal_width": 0.2
        }
    )

    assert response.status_code == 422


def test_invalid_string_input():
    response = client.post(
        "/predict",
        json={
            "sepal_length": "hello",
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
    )

    assert response.status_code == 422


def test_invalid_endpoint():
    response = client.get("/unknown")

    assert response.status_code == 404
