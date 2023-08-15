import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.model.train import (
    calc_class_weight,
    generate_metrics,
    train_model,
    treat_dataframe,
)


def test_treat_dataframe():
    df = pd.DataFrame(
        {
            "Unnamed: 32": [1, 2],
            "id": [3, 4],
            "concave points_mean": [5, 6],
            "other_column": [7, 8],
        }
    )
    treated_df = treat_dataframe(df)
    assert "Unnamed: 32" not in treated_df.columns
    assert "id" not in treated_df.columns
    assert "concave_points_mean" in treated_df.columns
    assert "other_column" in treated_df.columns


def test_calc_class_weight():
    y_train = pd.Series([0, 0, 1])
    y_test = pd.Series([0, 1])
    assert calc_class_weight(y_train, y_test) == 2.0


def test_train_model():
    X_train = pd.DataFrame({"feature": [1, 2, 3]})
    y_train = pd.Series([0, 1, 1])
    class_weight_ratio = 1.0
    model = train_model(X_train, y_train, class_weight_ratio)
    assert isinstance(model, RandomForestClassifier)


def test_generate_metrics():
    model = RandomForestClassifier()
    X_test = pd.DataFrame({"feature": [1, 2, 3]})
    y_test = pd.Series([0, 1, 1])
    model.fit(X_test, y_test)
    threshold = 0.5
    metrics = generate_metrics(model, threshold, X_test, y_test)
    assert "roc_auc_score" in metrics
