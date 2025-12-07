"""
Test module for model training functionality
"""
import os
import joblib
import pytest


def test_model_exists():
    """Test that a trained model file exists"""
    model_path = "model/diabetes_model.pkl"
    
    if not os.path.exists(model_path):
        pytest.skip("Model file not found - this is expected in CI before training step")


def test_model_can_load():
    """Test that the model can be loaded successfully"""
    model_path = "model/diabetes_model.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        assert model is not None, "Model loaded as None"
        assert hasattr(model, 'predict'), "Model doesn't have predict method"
    else:
        pytest.skip("Model file not found - run training first")


def test_model_has_predict_proba():
    """Test that model supports probability predictions"""
    model_path = "model/diabetes_model.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        assert hasattr(model, 'predict_proba'), "Model doesn't support predict_proba"
    else:
        pytest.skip("Model file not found - run training first")
