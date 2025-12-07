import os
import pytest
from src.load_data import load_data


def test_load_data():
    """Test that data can be loaded from the diabetes.csv file"""
    path = "data/diabetes.csv"
    
    if not os.path.exists(path):
        pytest.skip("Dataset file not found - this is expected in CI before DVC pull")

    df = load_data(path)

    # Check if dataframe is loaded
    assert df is not None
    assert len(df) > 0


def test_data_columns():
    """Test that loaded data has expected columns"""
    path = "data/diabetes.csv"
    
    if not os.path.exists(path):
        pytest.skip("Dataset file not found - this is expected in CI before DVC pull")
    
    df = load_data(path)
    
    expected_columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]
    
    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"


def test_data_no_nulls():
    """Test that critical columns don't have null values after loading"""
    path = "data/diabetes.csv"
    
    if not os.path.exists(path):
        pytest.skip("Dataset file not found - this is expected in CI before DVC pull")
    
    df = load_data(path)
    
    # Check that Outcome column has no nulls
    assert df['Outcome'].isnull().sum() == 0, "Outcome column has null values"
