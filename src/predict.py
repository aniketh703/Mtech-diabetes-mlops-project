import pandas as pd
import joblib
import os
import mlflow
import mlflow.sklearn
import numpy as np
from load_data import load_data
from mlflow_utils import get_mlflow_manager

def predict_with_mlflow(input_data_path="data/new_data.csv", use_latest_model=True, run_id=None):
    """
    Make predictions using MLflow tracked models
    
    Args:
        input_data_path: Path to input data CSV
        use_latest_model: If True, use the latest registered model
        run_id: Specific run ID to load model from (overrides use_latest_model)
    """
    
    # Initialize MLflow manager
    mlflow_manager = get_mlflow_manager()
    
    # -------- Step 1: Load the model --------
    if run_id:
        # Load model from specific run
        print(f"Loading model from run: {run_id}")
        model = mlflow_manager.load_model_from_run(run_id)
    elif use_latest_model:
        # Get the best model based on accuracy
        print("Loading best model based on test accuracy...")
        model, best_run_id = get_best_model_from_registry()
        if model is None:
            # Fallback to local model file
            print("No MLflow model found, falling back to local model file...")
            model = load_local_model()
            best_run_id = "local"
    else:
        # Load from local file
        model = load_local_model()
        best_run_id = "local"
    
    # -------- Step 2: Load new input data --------
    print(f"Loading input data from: {input_data_path}")
    df_new = load_data(input_data_path)
    
    # -------- Step 3: Make predictions --------
    predictions = model.predict(df_new)
    prediction_probabilities = model.predict_proba(df_new)
    
    # -------- Step 4: Log prediction run to MLflow --------
    with mlflow_manager.start_run(run_name="prediction_run"):
        # Log prediction metadata
        mlflow.log_param("input_data_path", input_data_path)
        mlflow.log_param("model_source", best_run_id)
        mlflow.log_param("num_predictions", len(predictions))
        mlflow.log_param("input_shape", df_new.shape)
        
        # Log prediction statistics
        positive_predictions = np.sum(predictions == 1)
        negative_predictions = np.sum(predictions == 0)
        
        mlflow.log_metric("positive_predictions", positive_predictions)
        mlflow.log_metric("negative_predictions", negative_predictions)
        mlflow.log_metric("positive_prediction_rate", positive_predictions / len(predictions))
        
        # Log average prediction confidence
        avg_confidence = np.mean(np.max(prediction_probabilities, axis=1))
        mlflow.log_metric("avg_prediction_confidence", avg_confidence)
        
        # Save prediction results
        results_df = df_new.copy()
        results_df['prediction'] = predictions
        results_df['diabetes_probability'] = prediction_probabilities[:, 1]
        results_df['confidence'] = np.max(prediction_probabilities, axis=1)
        
        # Save results to file
        results_path = "prediction_results.csv"
        results_df.to_csv(results_path, index=False)
        mlflow.log_artifact(results_path)
        
        print(f"MLflow prediction run completed: {mlflow.active_run().info.run_id}")
    
    # -------- Step 5: Print results --------
    print("\n=== Prediction Results ===")
    print(f"Total predictions: {len(predictions)}")
    print(f"Positive predictions (Diabetes): {positive_predictions}")
    print(f"Negative predictions (No Diabetes): {negative_predictions}")
    print(f"Average confidence: {avg_confidence:.4f}")
    
    print("\nDetailed predictions:")
    for i, (pred, prob, conf) in enumerate(zip(predictions, prediction_probabilities[:, 1], np.max(prediction_probabilities, axis=1))):
        result = "Diabetes" if pred == 1 else "No Diabetes"
        print(f"Sample {i+1}: {result} (Probability: {prob:.4f}, Confidence: {conf:.4f})")
    
    return predictions, prediction_probabilities, results_df

def get_best_model_from_registry():
    """Get the best model from MLflow model registry"""
    try:
        mlflow_manager = get_mlflow_manager()
        best_run = mlflow_manager.get_best_run(metric_name="test_accuracy")
        
        if best_run is not None:
            model = mlflow_manager.load_model_from_run(best_run['run_id'])
            return model, best_run['run_id']
        else:
            return None, None
    except Exception as e:
        print(f"Error loading model from registry: {e}")
        return None, None

def load_local_model():
    """Load model from local file system"""
    model_paths = [
        "artifacts/diabetes_model.pkl",
        "model/diabetes_model.pkl"
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"Loading local model from: {model_path}")
            return joblib.load(model_path)
    
    raise FileNotFoundError("No model file found. Train the model first.")

# Legacy function for backward compatibility
def predict_legacy():
    """Legacy prediction function using local model file"""
    # -------- Step 1: Load the saved model --------
    model = load_local_model()

    # -------- Step 2: Load new input data --------
    new_data_path = "data/new_data.csv"
    df_new = load_data(new_data_path)

    # -------- Step 3: Predict --------
    predictions = model.predict(df_new)

    # -------- Step 4: Print results --------
    print("Predictions for new data:")
    print(predictions)
    
    return predictions

if __name__ == "__main__":
    # Use MLflow-enhanced prediction by default
    try:
        predict_with_mlflow()
    except Exception as e:
        print(f"MLflow prediction failed: {e}")
        print("Falling back to legacy prediction...")
        predict_legacy()
