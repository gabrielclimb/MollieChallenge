from typing import Any

import joblib


def save_model(model: Any, path: str) -> None:
    return joblib.dump(model, path, compress="gzip")
