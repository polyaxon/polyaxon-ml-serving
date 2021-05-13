from typing import Dict

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

IRIS_CLASS_MAPPING = {0: "setosa", 1: "versicolor", 2: "virginica"}


def load_model(model_path: str):
    model = open(model_path, "rb")
    return joblib.load(model)


app = FastAPI()
classifier = load_model("./model.joblib")


class DataFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


def get_features(data: DataFeatures) -> np.ndarray:
    return np.array(
        [data.sepal_length, data.sepal_width, data.petal_length, data.petal_width],
        ndmin=2,
    )


def predict(features: np.ndarray, proba: bool = False) -> Dict:
    if proba:
        probabilities = {
            k: float(v)
            for k, v in zip(
                IRIS_CLASS_MAPPING.values(), classifier.predict_proba(features)[0]
            )
        }
        return {"probabilities": probabilities}

    prediction = int(classifier.predict(features)[0])
    return {
        "prediction": {"value": prediction, "class": IRIS_CLASS_MAPPING[prediction]}
    }


@app.post("/api/v1/predict")
async def get_prediction(data: DataFeatures):
    features = get_features(data)
    return predict(features)


@app.post("/api/v1/proba")
async def get_probabilities(data: DataFeatures):
    features = get_features(data)
    return predict(features, proba=True)


@app.get("/", response_class=HTMLResponse)
def index():
    return (
        "<p>Hello, This is a REST API used for Polyaxon ML Serving examples!</p>"
        "<p>Click the fullscreen button the get the URL of your serving API!<p/>"
    )
