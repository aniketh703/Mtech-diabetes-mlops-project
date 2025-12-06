"""
Test module for prediction functionality
"""
import os
import numpy as np
import pytest


def test_model_file():
    """Check model file exists in expected location"""
    # Model can be in either location
    model_paths = ["model/diabetes_model.pkl", "artifacts/diabetes_model.pkl"]
    model_exists = any(os.path.exists(path) for path in model_paths)
    assert model_exists, f"Model file not found in any of: {model_paths}"


def test_model_prediction():
    """Test that model can make predictions"""
    import joblib
    
    model_path = "model/diabetes_model.pkl"
    if not os.path.exists(model_path):
        pytest.skip("Model file not found")
    
    model = joblib.load(model_path)
    
    # Test with sample input (8 features for diabetes dataset)
    sample_input = np.array([[2, 120, 70, 20, 80, 25.0, 0.5, 30]])
    prediction = model.predict(sample_input)
    
    assert prediction is not None
    assert len(prediction) == 1
    assert prediction[0] in [0, 1]  # Binary classification


def test_model_probability():
    """Test that model can return probability predictions"""
    import joblib
    
    model_path = "model/diabetes_model.pkl"
    if not os.path.exists(model_path):
        pytest.skip("Model file not found")
    
    model = joblib.load(model_path)
    
    sample_input = np.array([[2, 120, 70, 20, 80, 25.0, 0.5, 30]])
    probabilities = model.predict_proba(sample_input)
    
    assert probabilities is not None
    assert probabilities.shape == (1, 2)  # 2 classes
    assert 0 <= probabilities[0][0] <= 1
    assert 0 <= probabilities[0][1] <= 1
