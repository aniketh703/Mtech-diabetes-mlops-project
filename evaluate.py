import json
import pandas as pd
import joblib
import os
import sys
import mlflow
import mlflow.sklearn
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate():
    data_path = "data/diabetes_processed.csv"
    model_path = "model/diabetes_model.pkl"
    
    # Check if required files exist (may not exist in CI before DVC pull)
    if not os.path.exists(data_path):
        print(f"Warning: Data file '{data_path}' not found.")
        print("This is expected in CI environments before DVC pull.")
        print("Skipping evaluation step.")
        return
    
    if not os.path.exists(model_path):
        print(f"Warning: Model file '{model_path}' not found.")
        print("This is expected in CI environments before training.")
        print("Skipping evaluation step.")
        return
    
    with open("mlflow_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    mlflow.set_tracking_uri(config["tracking_uri"])
    mlflow.set_experiment(config["experiment_name"])
    
    with mlflow.start_run(run_name=f"{config['run_name_prefix']}_evaluation"):
        
        df = pd.read_csv(data_path)
        
        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]
        
        mlflow.log_param("eval_dataset_shape", df.shape)
        mlflow.log_param("eval_samples", len(df))

        model = joblib.load(model_path)

        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1_score": f1_score(y, y_pred)
        }
        
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(f"eval_{metric_name}", metric_value)
        
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Diabetes', 'Diabetes'],
                    yticklabels=['No Diabetes', 'Diabetes'])
        plt.title('Confusion Matrix - Model Evaluation')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        confusion_matrix_path = "confusion_matrix.png"
        plt.savefig(confusion_matrix_path)
        mlflow.log_artifact(confusion_matrix_path)
        plt.close()
        
        class_report = classification_report(y, y_pred, output_dict=True)
        
        mlflow.log_metric("eval_class_0_precision", class_report['0']['precision'])
        mlflow.log_metric("eval_class_0_recall", class_report['0']['recall'])
        mlflow.log_metric("eval_class_0_f1", class_report['0']['f1-score'])
        mlflow.log_metric("eval_class_1_precision", class_report['1']['precision'])
        mlflow.log_metric("eval_class_1_recall", class_report['1']['recall'])
        mlflow.log_metric("eval_class_1_f1", class_report['1']['f1-score'])
        
        with open("classification_report.txt", "w") as f:
            f.write(classification_report(y, y_pred))
        mlflow.log_artifact("classification_report.txt")
        
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.hist(y_pred_proba[y == 0], bins=30, alpha=0.7, label='No Diabetes', color='blue')
        plt.hist(y_pred_proba[y == 1], bins=30, alpha=0.7, label='Diabetes', color='red')
        plt.xlabel('Prediction Probability')
        plt.ylabel('Count')
        plt.title('Prediction Probability Distribution')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.scatter(range(len(y_pred_proba)), y_pred_proba, c=y, alpha=0.6, cmap='coolwarm')
        plt.xlabel('Sample Index')
        plt.ylabel('Prediction Probability')
        plt.title('Prediction Probabilities by True Label')
        plt.colorbar(label='True Label')
        
        plt.tight_layout()
        pred_dist_path = "prediction_distribution.png"
        plt.savefig(pred_dist_path)
        mlflow.log_artifact(pred_dist_path)
        plt.close()

        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        
        mlflow.log_artifact("metrics.json")

        print("Evaluation complete. Metrics saved to metrics.json")
        print(f"Evaluation Results:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
        print(f"MLflow evaluation run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    evaluate()
