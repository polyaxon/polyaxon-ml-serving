from typing import Dict

import joblib
import numpy as np
from flask import Flask, jsonify, make_response, request


IRIS_CLASS_MAPPING = {0: "setosa", 1: "versicolor", 2: "virginica"}


def load_model(model_path: str):
    model = open(model_path, "rb")
    return joblib.load(model)


app = Flask(__name__)
classifier = load_model("./model.joblib")


def get_features(request_data: Dict[str, float]) -> np.ndarray:
    return np.array(
        [
            request_data["sepal_length"],
            request_data["sepal_width"],
            request_data["petal_length"],
            request_data["petal_width"],
        ],
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


@app.route("/api/v1/predict", methods=["POST"])
def get_prediction():
    request_data = request.json
    features = get_features(request_data)
    return make_response(jsonify(predict(features)))


@app.route("/api/v1/proba", methods=["POST"])
def get_probabilities():
    request_data = request.json
    features = get_features(request_data)
    return make_response(jsonify(predict(features, proba=True)))


@app.route("/", methods=["GET"])
def index():
    return (
        "<p>Hello, This is a REST API used for Polyaxon ML Serving examples!</p>"
        "<p>Click the fullscreen button the get the URL of your serving API!<p/>"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
