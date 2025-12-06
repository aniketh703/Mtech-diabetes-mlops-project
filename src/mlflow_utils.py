"""
MLflow utilities for diabetes prediction project
"""
import mlflow
import mlflow.sklearn
import yaml
import os
from pathlib import Path

class MLflowManager:
    def __init__(self, config_path="mlflow_config.yaml"):
        """Initialize MLflow manager with configuration"""
        self.config_path = config_path
        self.config = self.load_config()
        self.setup_mlflow()
    
    def load_config(self):
        """Load MLflow configuration from YAML file"""
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        else:
            # Default configuration if file doesn't exist
            return {
                "experiment_name": "diabetes_prediction",
                "tracking_uri": "file:./mlruns",
                "artifact_location": "./mlartifacts",
                "registered_model_name": "diabetes_model",
                "run_name_prefix": "diabetes_run"
            }
    
    def setup_mlflow(self):
        """Setup MLflow tracking URI and experiment"""
        mlflow.set_tracking_uri(self.config["tracking_uri"])
        
        # Create experiment if it doesn't exist
        try:
            mlflow.create_experiment(
                name=self.config["experiment_name"],
                artifact_location=self.config.get("artifact_location")
            )
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            pass
        
        mlflow.set_experiment(self.config["experiment_name"])
    
    def start_run(self, run_name=None, tags=None):
        """Start a new MLflow run with optional name and tags"""
        if run_name is None:
            run_name = f"{self.config['run_name_prefix']}_{mlflow.utils.time.get_current_time_millis()}"
        
        return mlflow.start_run(run_name=run_name, tags=tags or {})
    
    def log_model_info(self, model, model_name="model"):
        """Log model information and artifacts"""
        # Log model
        mlflow.sklearn.log_model(
            model,
            model_name,
            registered_model_name=self.config["registered_model_name"]
        )
        
        # Log model parameters if it's a sklearn model
        if hasattr(model, 'get_params'):
            params = model.get_params()
            for param_name, param_value in params.items():
                mlflow.log_param(f"model_{param_name}", param_value)
    
    def log_dataset_info(self, X, y=None, dataset_name="dataset"):
        """Log dataset information"""
        mlflow.log_param(f"{dataset_name}_shape", X.shape)
        mlflow.log_param(f"{dataset_name}_features", list(X.columns) if hasattr(X, 'columns') else X.shape[1])
        
        if y is not None:
            mlflow.log_param(f"{dataset_name}_target_shape", y.shape if hasattr(y, 'shape') else len(y))
            if hasattr(y, 'value_counts'):
                class_distribution = y.value_counts().to_dict()
                for class_label, count in class_distribution.items():
                    mlflow.log_param(f"{dataset_name}_class_{class_label}_count", count)
    
    def log_metrics_dict(self, metrics_dict, prefix=""):
        """Log a dictionary of metrics"""
        for metric_name, metric_value in metrics_dict.items():
            full_metric_name = f"{prefix}_{metric_name}" if prefix else metric_name
            mlflow.log_metric(full_metric_name, metric_value)
    
    def get_experiment_info(self):
        """Get current experiment information"""
        experiment = mlflow.get_experiment_by_name(self.config["experiment_name"])
        return {
            "experiment_id": experiment.experiment_id,
            "experiment_name": experiment.name,
            "lifecycle_stage": experiment.lifecycle_stage,
            "artifact_location": experiment.artifact_location
        }
    
    def list_runs(self, max_results=10):
        """List recent runs in the experiment"""
        experiment = mlflow.get_experiment_by_name(self.config["experiment_name"])
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=max_results,
            order_by=["start_time DESC"]
        )
        return runs
    
    def get_best_run(self, metric_name="test_accuracy", ascending=False):
        """Get the best run based on a specific metric"""
        runs = self.list_runs(max_results=100)
        if metric_name in runs.columns:
            best_run = runs.loc[runs[metric_name].idxmax() if not ascending else runs[metric_name].idxmin()]
            return best_run
        else:
            return None
    
    def load_model_from_run(self, run_id, model_name="model"):
        """Load a model from a specific run"""
        model_uri = f"runs:/{run_id}/{model_name}"
        return mlflow.sklearn.load_model(model_uri)
    
    def promote_model(self, run_id, stage="Production", model_name=None):
        """Promote a model to a specific stage in the model registry"""
        if model_name is None:
            model_name = self.config["registered_model_name"]
        
        client = mlflow.tracking.MlflowClient()
        
        # Get the model version
        model_version = client.get_latest_versions(
            model_name, 
            stages=["None", "Staging"]
        )
        
        if model_version:
            version = model_version[0].version
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            return f"Model {model_name} version {version} promoted to {stage}"
        else:
            return "No model version found to promote"

# Convenience function to get MLflow manager instance
def get_mlflow_manager(config_path="mlflow_config.yaml"):
    """Get an instance of MLflowManager"""
    return MLflowManager(config_path)