import os
import joblib
import mlflow
import mlflow.sklearn
import yaml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.load_data import load_data

def train_model():
    # Load MLflow configuration
    with open("mlflow_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Set MLflow tracking URI and experiment
    mlflow.set_tracking_uri(config["tracking_uri"])
    mlflow.set_experiment(config["experiment_name"])
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"{config['run_name_prefix']}_src_training"):
        
        # -------- Step 1: Load the dataset --------
        data_path = "data/diabetes.csv"
        df = load_data(data_path)
        
        # Log dataset information
        mlflow.log_param("data_source", data_path)
        mlflow.log_param("dataset_shape", df.shape)
        mlflow.log_param("dataset_columns", list(df.columns))

        # -------- Step 2: Split features and target --------
        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]
        
        # Log feature information
        mlflow.log_param("num_features", len(X.columns))
        mlflow.log_param("features", list(X.columns))
        mlflow.log_param("target", "Outcome")

        # -------- Step 3: Train-test split --------
        test_size = 0.2
        random_state = 42
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Log split parameters
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))

        # -------- Step 4: Train model --------
        max_iter = 1000
        model = LogisticRegression(max_iter=max_iter, random_state=random_state)
        
        # Log model hyperparameters
        mlflow.log_param("algorithm", "LogisticRegression")
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("solver", model.solver)
        mlflow.log_param("penalty", model.penalty)
        mlflow.log_param("C", model.C)
        
        model.fit(X_train, y_train)

        # -------- Step 5: Evaluate model --------
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate comprehensive metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        
        # Log all metrics
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1_score", test_f1)
        
        print(f"Model Accuracy: {test_accuracy:.4f}")
        print(f"Model Precision: {test_precision:.4f}")
        print(f"Model Recall: {test_recall:.4f}")
        print(f"Model F1-Score: {test_f1:.4f}")

        # -------- Step 6: Create artifacts folder --------
        os.makedirs("artifacts", exist_ok=True)

        # -------- Step 7: Save model --------
        model_path = "artifacts/diabetes_model.pkl"
        joblib.dump(model, model_path)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name=f"{config['registered_model_name']}_src"
        )
        
        # Log model artifact
        mlflow.log_artifact(model_path, "local_artifacts")
        
        print(f"Model saved at: {model_path}")
        print(f"MLflow run completed with ID: {mlflow.active_run().info.run_id}")
        
        return model, test_accuracy

if __name__ == "__main__":
    train_model()
