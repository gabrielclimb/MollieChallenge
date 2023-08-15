from typing import Any, Tuple

import joblib


def save_model(model: Any, path: str) -> None:
    """
    Saves the provided model to the specified path.

    Args:
        model (Any): The model object to be saved.
        path (str): The path where the model will be saved, including the filename.

    Returns:
        None: The function doesn't return anything; it saves the model to disk.
    """
    return joblib.dump(model, path, compress="gzip")


def load_model_and_threshold(path: str) -> Tuple[Any, float]:
    """
    Loads a model and associated threshold from the specified path.

    Args:
        path (str): The path from where the model and threshold will be loaded,
            including the filename.

    Returns:
        Tuple[Any, float]: A tuple containing the loaded model object and associated
            threshold.
    """
    return joblib.load(path)
