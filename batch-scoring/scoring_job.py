import argparse

import joblib
import pandas as pd

from polyaxon import tracking
from polyaxon.polyboard.artifacts import V1ArtifactKind

IRIS_CLASS_MAPPING = {0: "setosa", 1: "versicolor", 2: "virginica"}


def load_model(model_path: str):
    model = open(model_path, "rb")
    return joblib.load(model)


def load_dataset(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def score(data: pd.DataFrame) -> pd.DataFrame:
    feature_columns = ["sepal.length", "sepal.width", "petal.length", "petal.width"]
    data['prediction'] = classifier.predict(data[feature_columns].values)
    data['prediction_class'] = (data['prediction'].apply(lambda i: IRIS_CLASS_MAPPING[i]))

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-path',
        type=str,
        default="./model.joblib",
    )
    parser.add_argument(
        '--csv-path',
        type=str,
        default="./inputs.csv",
    )
    args = parser.parse_args()

    tracking.init()

    classifier = load_model(args.model_path)
    print("Started scoring csv {}!".format(args.csv_path))
    data = load_dataset(args.csv_path)
    scores = score(data)

    results_path = tracking.get_outputs_path("results.csv")
    scores.to_csv(results_path, index=False)
    tracking.log_artifact_ref(
        results_path,
        name="scoring-results",
        is_input=False,
        kind=V1ArtifactKind.CSV,
    )
    print("Finished scoring!")
