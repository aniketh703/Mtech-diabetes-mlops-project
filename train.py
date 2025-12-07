import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import sys
import mlflow
import mlflow.sklearn
import yaml

with open("mlflow_config.yaml", "r") as f:
    config = yaml.safe_load(f)

mlflow.set_tracking_uri(config["tracking_uri"])
mlflow.set_experiment(config["experiment_name"])

data_path = "data/diabetes_processed.csv"
model_dir = "model"
model_path = os.path.join(model_dir, "diabetes_model.pkl")

# Check if processed data file exists (may not exist in CI before preprocessing)
if not os.path.exists(data_path):
    print(f"Warning: Processed data file '{data_path}' not found.")
    print("This is expected in CI environments before DVC pull/preprocessing.")
    print("Skipping training step.")
    sys.exit(0)

os.makedirs(model_dir, exist_ok=True)

with mlflow.start_run(run_name=f"{config['run_name_prefix']}_main_training"):
    
    df = pd.read_csv(data_path)
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    
    mlflow.log_param("dataset_shape", df.shape)
    mlflow.log_param("features", list(X.columns))
    mlflow.log_param("target_column", "Outcome")

    test_size = 0.2
    random_state = 42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("train_samples", len(X_train))
    mlflow.log_param("test_samples", len(X_test))

    max_iter = 200
    model = LogisticRegression(max_iter=max_iter, random_state=random_state)
    
    mlflow.log_param("algorithm", "LogisticRegression")
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_param("solver", model.solver)
    
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("test_recall", test_recall)
    mlflow.log_metric("test_f1_score", test_f1)
    
    mlflow.sklearn.log_model(
        model, 
        "model",
        registered_model_name=config["registered_model_name"]
    )
    
    joblib.dump(model, model_path)
    
    mlflow.log_artifact(model_path, "local_model")
    
    print(f"Model saved to {model_path}")
    print(f"MLflow run completed with accuracy: {test_accuracy:.4f}")
    print(f"MLflow run ID: {mlflow.active_run().info.run_id}")
