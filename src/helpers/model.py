from typing import Any, Tuple

import joblib


def save_model(model: Any, path: str) -> None:
    return joblib.dump(model, path, compress="gzip")


def load_model_and_threshold(path: str) -> Tuple[Any, float]:
    return joblib.load(path)
