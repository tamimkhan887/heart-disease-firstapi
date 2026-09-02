from fastapi import FastAPI
import joblib
import pandas as pd
from pathlib import Path

from .schemas import HeartDiseaseInput


BASE_DIR = Path(__file__).resolve().parent.parent
model = joblib.load(BASE_DIR / "model" / "heart_model.joblib")


# Feature order
FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]


# Create FastAPI application
app = FastAPI(
    title="Heart Disease Prediction API",
    description="FastAPI service for heart disease prediction",
    version="1.0.0"
)


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/info")
def info():
    return {
        "model": "Random Forest Classifier",
        "features": FEATURES,
        "number_of_features": len(FEATURES)
    }


@app.post("/predict")
def predict(data: HeartDiseaseInput):

    input_data = pd.DataFrame(
        [[
            data.age,
            data.sex,
            data.cp,
            data.trestbps,
            data.chol,
            data.fbs,
            data.restecg,
            data.thalach,
            data.exang,
            data.oldpeak,
            data.slope,
            data.ca,
            data.thal
        ]],
        columns=FEATURES
    )

    prediction = model.predict(input_data)[0]

    return {
        "heart_disease": bool(prediction == 1)
    }