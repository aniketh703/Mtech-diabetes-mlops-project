from load_data import load_data
from train import model, accuracy
import pandas as pd
import joblib
import os

def run_pipeline():

    print("\n===== STEP 1: DATA LOADING =====")
    df = load_data("data/diabetes.csv")
    print("Data loaded successfully.")

    print("\n===== STEP 2: MODEL TRAINING =====")
    print(f"Model already trained with accuracy: {accuracy}")

    print("\n===== STEP 3: SAVING MODEL =====")
    os.makedirs("artifacts", exist_ok=True)
    model_path = "artifacts/diabetes_model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved at: {model_path}")

    print("\n===== STEP 4: PREDICTING ON NEW DATA =====")
    new_df = load_data("data/new_data.csv")
    predictions = model.predict(new_df)
    print("Predictions on new data:")
    print(predictions)

if __name__ == "__main__":
    run_pipeline()
