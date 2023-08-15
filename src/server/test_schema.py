from src.server.schemas import BreastCancerData
import pytest
from pydantic import ValidationError


def test_breast_cancer_schema():
    data = {
        "radius_mean": 20.18,
        "texture_mean": 23.97,
        "perimeter_mean": 143.7,
        "area_mean": 1245,
        "smoothness_mean": 0.1286,
        "compactness_mean": 0.3454,
        "concavity_mean": 0.3754,
        "concave_points_mean": 0.1604,
        "symmetry_mean": 0.2906,
        "fractal_dimension_mean": 0.08142,
        "radius_se": 0.9317,
        "texture_se": 1.885,
        "perimeter_se": 8.649,
        "area_se": 116.4,
        "smoothness_se": 0.01038,
        "compactness_se": 0.06835,
        "concavity_se": 0.1091,
        "concave_points_se": 0.02593,
        "symmetry_se": 0.07895,
        "fractal_dimension_se": 0.005987,
        "radius_worst": 23.37,
        "texture_worst": 31.72,
        "perimeter_worst": 170.3,
        "area_worst": 1623,
        "smoothness_worst": 0.1639,
        "compactness_worst": 0.6164,
        "concavity_worst": 0.7681,
        "concave_points_worst": 0.2508,
        "symmetry_worst": 0.544,
        "fractal_dimension_worst": 0.09964,
    }
    schema = BreastCancerData(**data)
    assert schema.radius_mean == 20.18


def test_breast_cancer_wrong_schema():
    wrong_data = {"field": 0.02}
    with pytest.raises(ValidationError):
        _ = BreastCancerData(**wrong_data)
