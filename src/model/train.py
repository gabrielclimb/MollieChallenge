from typing import Any, Dict

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from yellowbrick.classifier import DiscriminationThreshold


def main():
    dataset_url = "https://raw.githubusercontent.com/pkmklong/Breast-Cancer-Wisconsin-Diagnostic-DataSet/master/data.csv"
    df = pd.read_csv(dataset_url)
    df = treat_dataframe(df)
    y_column_name = "diagnosis"
    y = y = df["diagnosis"].map({"B": 0, "M": 1})
    X = df.drop(columns=[y_column_name])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.3, stratify=y
    )

    class_weight_ratio = calc_class_weight(y_train, y_test)

    rf_model = train_model(X_train, y_train, class_weight_ratio)

    best_threshold = get_best_threshold(rf_model, class_weight_ratio, X_train, y_train)

    log_model_mlflow(X_train, X_test, y_test, rf_model, best_threshold)


def treat_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    columns_with_no_meaning = ["Unnamed: 32", "id"]
    columns_to_rename = {
        "concave points_mean": "concave_points_mean",
        "concave points_se": "concave_points_se",
        "concave points_worst": "concave_points_worst",
    }

    df = df.drop(columns=columns_with_no_meaning)
    df = df.rename(columns=columns_to_rename)
    return df


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
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def plot_confusion_matrix(y_test, y_pred):
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, y_pred, normalize="true") * 100
    df = pd.DataFrame(cm.T, index=["B", "M"], columns=["B", "M"])
    ax = sns.heatmap(df, annot=True, cmap=sns.color_palette("Blues"))
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    return fig


def plot_precision_recall(precisions, recalls, thresholds):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(thresholds, precisions[:-1], "r--", label="Precisions")
    ax.plot(thresholds, recalls[:-1], "#424242", label="Recalls")
    ax.set_title("Precision and Recall \n Tradeoff", fontsize=18)
    ax.set_ylabel("Level of Precision and Recall", fontsize=16)
    ax.set_xlabel("Thresholds", fontsize=16)
    ax.legend(loc="best", fontsize=14)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    return fig


def log_model_mlflow(X_train, X_test, y_test, rf_model, best_threshold):
    experiment_name = "Breast-Cancer-Training"
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name="BreastCancerModelTraining"):
        y_prob = rf_model.predict_proba(X_test)[:, 1]

        mlflow.sklearn.log_model(
            rf_model,
            artifact_path="model",
            input_example=X_train.head(),
            pyfunc_predict_fn="predict_proba",
        )

        mlflow.log_params(rf_model.get_params())
        mlflow.log_param("threshold", best_threshold)
        mlflow.log_metrics(generate_metrics(rf_model, best_threshold, X_test, y_test))

        precision_rf, recall_rf, thresholds_rf = precision_recall_curve(y_test, y_prob)

        threshold = best_threshold
        y_pred = y_prob > threshold
        plot_confusion_matrix(y_test, y_pred)

        mlflow.log_figure(
            plot_confusion_matrix(y_test, y_pred),
            "confusion_matrix.png",
        )

        mlflow.log_figure(
            plot_precision_recall(precision_rf, recall_rf, thresholds_rf),
            "precision_recall.png",
        )


if __name__ == "__main__":
    main()
