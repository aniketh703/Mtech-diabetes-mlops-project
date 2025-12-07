"""
Prefect Workflow Orchestration for Diabetes MLOps Pipeline
===========================================================
This module implements a complete ML pipeline using Prefect for workflow orchestration.
It includes tasks for data preprocessing, model training, and evaluation with proper
error handling, retries, and logging.
"""

import os
import json
import logging
from datetime import timedelta
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash

# Configure logging
logging.basicConfig(level=logging.INFO)


# =============================================================================
# DATA PREPROCESSING TASKS
# =============================================================================

@task(
    name="load_raw_data",
    description="Load raw diabetes dataset from CSV file",
    retries=2,
    retry_delay_seconds=10,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1)
)
def load_raw_data(data_path: str = "data/diabetes.csv") -> pd.DataFrame:
    """
    Load raw diabetes data from CSV file.
    
    Args:
        data_path: Path to the raw data file
        
    Returns:
        DataFrame with raw data
    """
    logger = get_run_logger()
    logger.info(f"Loading raw data from {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df


@task(
    name="validate_data",
    description="Validate data quality and check for issues",
    retries=1
)
def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate data quality and return validation report.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with validation results
    """
    logger = get_run_logger()
    logger.info("Validating data quality...")
    
    validation_report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": df.duplicated().sum(),
        "dtypes": df.dtypes.astype(str).to_dict()
    }
    
    # Check for zero values in columns that shouldn't have zeros
    zero_invalid_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    zero_counts = {}
    for col in zero_invalid_cols:
        if col in df.columns:
            zero_counts[col] = (df[col] == 0).sum()
    validation_report["zero_values"] = zero_counts
    
    logger.info(f"Found {validation_report['duplicates']} duplicate rows")
    logger.info(f"Zero values in key columns: {zero_counts}")
    
    return validation_report


@task(
    name="handle_missing_values",
    description="Handle missing values and invalid zeros using median imputation",
    retries=1
)
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values and replace invalid zeros with median values.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with handled missing values
    """
    logger = get_run_logger()
    logger.info("Handling missing values...")
    
    df_clean = df.copy()
    
    # Remove duplicates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    logger.info(f"Removed {initial_rows - len(df_clean)} duplicate rows")
    
    # Replace zeros with NaN for columns where zero is invalid
    zero_invalid_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_invalid_cols:
        if col in df_clean.columns:
            df_clean.loc[df_clean[col] == 0, col] = np.nan
    
    # Impute missing values with median
    target_col = "Outcome"
    feature_cols = [col for col in df_clean.columns if col != target_col]
    
    imputer = SimpleImputer(strategy="median")
    df_clean[feature_cols] = imputer.fit_transform(df_clean[feature_cols])
    
    logger.info(f"Imputed missing values. Final shape: {df_clean.shape}")
    
    return df_clean


@task(
    name="feature_engineering",
    description="Create additional features and prepare data for modeling"
)
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    logger = get_run_logger()
    logger.info("Performing feature engineering...")
    
    df_featured = df.copy()
    
    # Create age groups
    if "Age" in df_featured.columns:
        df_featured["AgeGroup"] = pd.cut(
            df_featured["Age"],
            bins=[0, 30, 45, 60, 100],
            labels=[0, 1, 2, 3]
        ).astype(int)
    
    # Create BMI categories
    if "BMI" in df_featured.columns:
        df_featured["BMICategory"] = pd.cut(
            df_featured["BMI"],
            bins=[0, 18.5, 25, 30, 100],
            labels=[0, 1, 2, 3]
        ).astype(int)
    
    # Create glucose level indicator
    if "Glucose" in df_featured.columns:
        df_featured["HighGlucose"] = (df_featured["Glucose"] > 140).astype(int)
    
    logger.info(f"Feature engineering complete. New columns added. Shape: {df_featured.shape}")
    
    return df_featured


@task(
    name="save_processed_data",
    description="Save processed data to CSV file"
)
def save_processed_data(df: pd.DataFrame, output_path: str = "data/diabetes_processed.csv") -> str:
    """
    Save processed data to CSV file.
    
    Args:
        df: Processed DataFrame
        output_path: Path to save the processed data
        
    Returns:
        Path to saved file
    """
    logger = get_run_logger()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Saved processed data to {output_path}")
    return output_path


# =============================================================================
# MODEL TRAINING TASKS
# =============================================================================

@task(
    name="prepare_train_test_split",
    description="Split data into training and testing sets"
)
def prepare_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.
    
    Args:
        df: Input DataFrame
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger = get_run_logger()
    logger.info(f"Splitting data with test_size={test_size}")
    
    target_col = "Outcome"
    
    # Use only original features for training (not engineered ones for simplicity)
    feature_cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Testing set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test


@task(
    name="scale_features",
    description="Standardize features using StandardScaler"
)
def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Scale features using StandardScaler.
    
    Args:
        X_train: Training features
        X_test: Testing features
        
    Returns:
        Tuple of (scaled X_train, scaled X_test, fitted scaler)
    """
    logger = get_run_logger()
    logger.info("Scaling features...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info("Feature scaling complete")
    return X_train_scaled, X_test_scaled, scaler


@task(
    name="train_model",
    description="Train logistic regression model",
    retries=2,
    retry_delay_seconds=5
)
def train_model(
    X_train: np.ndarray,
    y_train: pd.Series,
    max_iter: int = 1000,
    random_state: int = 42
) -> LogisticRegression:
    """
    Train a logistic regression model.
    
    Args:
        X_train: Training features (scaled)
        y_train: Training labels
        max_iter: Maximum iterations for convergence
        random_state: Random seed
        
    Returns:
        Trained model
    """
    logger = get_run_logger()
    logger.info("Training logistic regression model...")
    
    model = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        solver='lbfgs',
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    train_accuracy = model.score(X_train, y_train)
    logger.info(f"Training accuracy: {train_accuracy:.4f}")
    
    return model


@task(
    name="save_model",
    description="Save trained model to disk"
)
def save_model(model: LogisticRegression, model_path: str = "model/diabetes_model.pkl") -> str:
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        model_path: Path to save the model
        
    Returns:
        Path to saved model
    """
    logger = get_run_logger()
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    
    logger.info(f"Model saved to {model_path}")
    return model_path


# =============================================================================
# MODEL EVALUATION TASKS
# =============================================================================

@task(
    name="evaluate_model",
    description="Evaluate model performance on test set"
)
def evaluate_model(
    model: LogisticRegression,
    X_test: np.ndarray,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_test: Test features (scaled)
        y_test: Test labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger = get_run_logger()
    logger.info("Evaluating model performance...")
    
    y_pred = model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }
    
    # Log confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")
    
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")
    
    return metrics


@task(
    name="save_metrics",
    description="Save evaluation metrics to JSON file"
)
def save_metrics(metrics: Dict[str, float], metrics_path: str = "metrics.json") -> str:
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        metrics_path: Path to save metrics
        
    Returns:
        Path to saved metrics file
    """
    logger = get_run_logger()
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Metrics saved to {metrics_path}")
    return metrics_path


@task(
    name="generate_report",
    description="Generate final pipeline execution report"
)
def generate_report(
    validation_report: Dict[str, Any],
    metrics: Dict[str, float],
    model_path: str,
    data_path: str
) -> Dict[str, Any]:
    """
    Generate final pipeline execution report.
    
    Args:
        validation_report: Data validation results
        metrics: Model evaluation metrics
        model_path: Path to saved model
        data_path: Path to processed data
        
    Returns:
        Complete pipeline report
    """
    logger = get_run_logger()
    
    report = {
        "pipeline_status": "SUCCESS",
        "data_validation": validation_report,
        "model_metrics": metrics,
        "artifacts": {
            "model_path": model_path,
            "processed_data_path": data_path
        }
    }
    
    logger.info("=" * 50)
    logger.info("PIPELINE EXECUTION REPORT")
    logger.info("=" * 50)
    logger.info(f"Status: {report['pipeline_status']}")
    logger.info(f"Model Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Model F1 Score: {metrics['f1_score']:.4f}")
    logger.info("=" * 50)
    
    return report


# =============================================================================
# MAIN PIPELINE FLOWS
# =============================================================================

@flow(
    name="data_preprocessing_flow",
    description="Data preprocessing sub-flow",
    retries=1
)
def data_preprocessing_flow(data_path: str = "data/diabetes.csv") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Data preprocessing sub-flow.
    
    Args:
        data_path: Path to raw data
        
    Returns:
        Tuple of (processed DataFrame, validation report)
    """
    # Load data
    df_raw = load_raw_data(data_path)
    
    # Validate data
    validation_report = validate_data(df_raw)
    
    # Handle missing values
    df_clean = handle_missing_values(df_raw)
    
    # Feature engineering
    df_processed = feature_engineering(df_clean)
    
    # Save processed data
    save_processed_data(df_processed)
    
    return df_processed, validation_report


@flow(
    name="model_training_flow",
    description="Model training sub-flow",
    retries=1
)
def model_training_flow(df: pd.DataFrame) -> Tuple[LogisticRegression, np.ndarray, pd.Series, str]:
    """
    Model training sub-flow.
    
    Args:
        df: Processed DataFrame
        
    Returns:
        Tuple of (trained model, X_test_scaled, y_test, model_path)
    """
    # Split data
    X_train, X_test, y_train, y_test = prepare_train_test_split(df)
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Train model
    model = train_model(X_train_scaled, y_train)
    
    # Save model
    model_path = save_model(model)
    
    return model, X_test_scaled, y_test, model_path


@flow(
    name="model_evaluation_flow",
    description="Model evaluation sub-flow"
)
def model_evaluation_flow(
    model: LogisticRegression,
    X_test: np.ndarray,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Model evaluation sub-flow.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save metrics
    save_metrics(metrics)
    
    return metrics


@flow(
    name="diabetes_ml_pipeline",
    description="Complete end-to-end ML pipeline for diabetes prediction",
    version="1.0.0"
)
def diabetes_ml_pipeline(
    data_path: str = "data/diabetes.csv",
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Complete end-to-end ML pipeline for diabetes prediction.
    
    This flow orchestrates the entire ML workflow:
    1. Data preprocessing and validation
    2. Model training
    3. Model evaluation
    4. Report generation
    
    Args:
        data_path: Path to raw data file
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Complete pipeline execution report
    """
    logger = get_run_logger()
    logger.info("Starting Diabetes ML Pipeline...")
    logger.info(f"Configuration: data_path={data_path}, test_size={test_size}")
    
    # Step 1: Data Preprocessing
    logger.info("Step 1: Data Preprocessing")
    df_processed, validation_report = data_preprocessing_flow(data_path)
    
    # Step 2: Model Training
    logger.info("Step 2: Model Training")
    model, X_test_scaled, y_test, model_path = model_training_flow(df_processed)
    
    # Step 3: Model Evaluation
    logger.info("Step 3: Model Evaluation")
    metrics = model_evaluation_flow(model, X_test_scaled, y_test)
    
    # Step 4: Generate Report
    logger.info("Step 4: Generating Report")
    report = generate_report(
        validation_report=validation_report,
        metrics=metrics,
        model_path=model_path,
        data_path="data/diabetes_processed.csv"
    )
    
    logger.info("Pipeline completed successfully!")
    return report


# =============================================================================
# SCHEDULED/TRIGGERED FLOWS
# =============================================================================

@flow(
    name="retrain_pipeline",
    description="Retraining pipeline for model updates"
)
def retrain_pipeline(data_path: str = "data/new_data.csv") -> Dict[str, Any]:
    """
    Retraining pipeline for when new data is available.
    
    Args:
        data_path: Path to new data
        
    Returns:
        Pipeline execution report
    """
    logger = get_run_logger()
    logger.info("Starting model retraining pipeline...")
    
    # Use main pipeline with new data path
    if os.path.exists(data_path):
        return diabetes_ml_pipeline(data_path=data_path)
    else:
        logger.warning(f"New data file not found: {data_path}")
        logger.info("Using default training data")
        return diabetes_ml_pipeline()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Run the main pipeline
    print("=" * 60)
    print("DIABETES MLOPS PIPELINE - PREFECT WORKFLOW")
    print("=" * 60)
    
    # Execute the pipeline
    result = diabetes_ml_pipeline()
    
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 60)
    print(f"Status: {result['pipeline_status']}")
    print(f"Model Accuracy: {result['model_metrics']['accuracy']:.4f}")
    print(f"Model F1 Score: {result['model_metrics']['f1_score']:.4f}")
    print("=" * 60)
