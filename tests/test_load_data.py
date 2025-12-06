import os
from src.load_data import load_data

def test_load_data():
    path = "data/diabetes.csv"
    
    assert os.path.exists(path), "Dataset file missing!"

    df = load_data(path)

    # Check if dataframe is loaded
    assert df is not None
    assert len(df) > 0
