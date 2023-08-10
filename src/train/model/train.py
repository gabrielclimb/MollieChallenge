from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    train_test_split,
)
from yellowbrick.classifier import DiscriminationThreshold

from src.train.helpers.model import save_model


def main():
    columns_with_no_meaning = ["Unnamed: 32", "id"]
    y_column_name = "diagnosis"
    df = pd.read_csv("cancer.csv").drop(columns=columns_with_no_meaning)

    y = y = df["diagnosis"].map({"B": 0, "M": 1})
    X = df.drop(columns=[y_column_name])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.3, stratify=y
    )
    # TODO: Save data
    class_weight_ratio = calc_class_weight(y_train, y_test)
    rf_model = train_model(X_train, y_train, class_weight_ratio)

    best_threshold = get_best_threshold(rf_model, class_weight_ratio, X_train, y_train)

    save_model(rf_model, f"src/artifacts/model_{best_threshold}.joblib")

    print(generate_metrics(rf_model, best_threshold, X_test, y_test))


def calc_class_weight(y_train: pd.Series, y_test: pd.Series) -> np.float64:
    class_distribution_train = y_train.value_counts(normalize=True)
    return class_distribution_train[0] / class_distribution_train[1]


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, class_weight_ratio: float
) -> RandomForestClassifier:
    rf_model = RandomForestClassifier(
        class_weight={0: 1, 1: class_weight_ratio}, random_state=42
    )

    return rf_model.fit(X_train, y_train)


def get_best_threshold(
    model: Any, class_weight_ratio: float, X_train: pd.DataFrame, y_train: pd.Series
) -> float:
    thresholds = DiscriminationThreshold(
        model,
        quantiles=np.array([0.0, 0.5, 0.75]),
        argmax="fscore",
        exclude="queue_rate",
        fbeta=class_weight_ratio,
    )

    thresholds.fit(X_train, y_train)
    best_threshold = thresholds.thresholds_[
        thresholds.cv_scores_[thresholds.argmax].argmax()
    ]
    return best_threshold


def generate_metrics(
    model: RandomForestClassifier,
    threshold: float,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    y_prob = model.predict_proba(X_test)[:, 1] > threshold
    y_pred = model.predict(X_test)

    tn, fp, fn, tp = (confusion_matrix(y_test, y_pred, normalize="true") * 100).ravel()
    return {
        "roc_auc_score": roc_auc_score(y_test, y_prob),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


if __name__ == "__main__":
    main()
