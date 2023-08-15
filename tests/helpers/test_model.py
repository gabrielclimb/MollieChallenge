import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.helpers.model import save_model, load_model_and_threshold


def test_save_and_load_model_and_threshold(tmp_path):
    model = LogisticRegression()
    model.fit(np.array([[1], [2]]), np.array([0, 1]))

    model_path = str(tmp_path / "model.joblib")
    save_model((model, 0.5), model_path)

    assert os.path.exists(model_path)

    loaded_model, threshold = load_model_and_threshold(model_path)

    assert isinstance(loaded_model, LogisticRegression)
    assert threshold == 0.5
