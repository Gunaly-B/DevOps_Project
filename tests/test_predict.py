"""
Unit tests for inference predictor module.
"""

import pytest
from src.schemas import HeartDiseaseInput
from src.predict import HeartDiseasePredictor


@pytest.fixture
def sample_input():
    return HeartDiseaseInput(
        age=57.0,
        sex=1.0,
        cp=4.0,
        trestbps=140.0,
        chol=241.0,
        fbs=0.0,
        restecg=1.0,
        thalach=123.0,
        exang=1.0,
        oldpeak=1.5,
        slope=2.0,
        ca=0.0,
        thal=7.0,
    )


def test_predictor_execution(sample_input):
    predictor = HeartDiseasePredictor()
    if predictor.model is None:
        pytest.skip("Model not yet trained. Run training first.")

    res = predictor.predict(sample_input)
    assert res.prediction in [0, 1]
    assert 0.0 <= res.probability <= 1.0
    assert isinstance(res.heart_disease_risk, str)
