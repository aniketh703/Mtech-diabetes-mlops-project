# MLflow Integration Guide for Diabetes Prediction Project

## Overview

This project now includes comprehensive MLflow tracking for experiment management, model versioning, and deployment. MLflow provides centralized tracking of model parameters, metrics, artifacts, and model registry functionality.

## Features Implemented

### 1. **Experiment Tracking**
- **Parameters**: All hyperparameters (learning rate, regularization, etc.)
- **Metrics**: Accuracy, precision, recall, F1-score for both training and testing
- **Artifacts**: Model files, plots, confusion matrices, classification reports
- **Dataset Info**: Shape, features, class distributions

### 2. **Model Registry**
- Automatic model registration with versioning
- Model staging (Development → Staging → Production)
- Model comparison and promotion capabilities

### 3. **Visualization**
- Confusion matrices as artifacts
- Prediction probability distributions
- Model performance plots

### 4. **Integration with DVC**
- MLflow tracking works alongside existing DVC pipeline
- Metrics and plots are tracked by both systems

## Setup and Configuration

### 1. **Dependencies**
All required packages are listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. **Configuration**
MLflow settings are defined in `mlflow_config.yaml`:
```yaml
experiment_name: "diabetes_prediction"
tracking_uri: "file:./mlruns"
artifact_location: "./mlartifacts"
registered_model_name: "diabetes_model"
run_name_prefix: "diabetes_run"
```

## Usage Guide

### 1. **Training with MLflow Tracking**

#### Standard Training (train.py):
```bash
python train.py
```
This will:
- Log all hyperparameters
- Track training and validation metrics
- Save model to MLflow registry
- Create visualizations

#### Source Training (src/train.py):
```bash
python src/train.py
```
Similar functionality with different data loading approach.

### 2. **Model Evaluation**
```bash
python evaluate.py
```
This will:
- Log evaluation metrics
- Generate confusion matrix
- Create prediction distribution plots
- Save classification report

### 3. **Making Predictions**
```bash
python src/predict.py
```
Enhanced prediction with MLflow integration:
- Automatically loads best model from registry
- Logs prediction runs
- Saves detailed results with confidence scores

### 4. **MLflow UI**
Start the MLflow tracking UI:
```bash
python mlflow_manager.py ui
```
Or directly:
```bash
mlflow ui --host 0.0.0.0 --port 5000
```
Access at: http://localhost:5000

### 5. **Experiment Management**

#### List All Experiments:
```bash
python mlflow_manager.py list-experiments
```

#### List Recent Runs:
```bash
python mlflow_manager.py list-runs --max-results 20
```

#### Compare Specific Runs:
```bash
python mlflow_manager.py compare run1_id run2_id run3_id
```

#### Get Best Performing Model:
```bash
python mlflow_manager.py best-model --metric test_f1_score
```

#### Promote Model to Production:
```bash
python mlflow_manager.py promote <run_id> --stage Production
```

#### Cleanup Old Runs:
```bash
python mlflow_manager.py cleanup --keep 20
```

## File Structure

```
diabetes-mlops-project/
├── mlflow_config.yaml          # MLflow configuration
├── mlflow_manager.py           # Experiment management utilities
├── requirements.txt            # Updated with MLflow dependencies
├── train.py                    # Enhanced with MLflow tracking
├── evaluate.py                 # Enhanced with MLflow tracking
├── src/
│   ├── mlflow_utils.py         # MLflow utility classes
│   ├── train.py                # Enhanced training script
│   └── predict.py              # Enhanced prediction with MLflow
├── mlruns/                     # MLflow tracking store (created automatically)
└── mlartifacts/                # MLflow artifacts store (created automatically)
```

## Key MLflow Features Utilized

### 1. **Parameter Logging**
- Dataset information (shape, features, splits)
- Model hyperparameters (solver, regularization, iterations)
- Training configuration (test size, random state)

### 2. **Metric Tracking**
- Training accuracy
- Test accuracy, precision, recall, F1-score
- Class-specific metrics
- Prediction confidence metrics

### 3. **Artifact Management**
- Model files (.pkl)
- Visualization plots (.png)
- Classification reports (.txt)
- Prediction results (.csv)

### 4. **Model Registry**
- Automatic model registration
- Version management
- Stage transitions (None → Staging → Production)
- Model comparison capabilities

## Best Practices Implemented

1. **Consistent Naming**: All runs follow naming conventions
2. **Comprehensive Logging**: Parameters, metrics, and artifacts are systematically logged
3. **Model Versioning**: Automatic registration and versioning
4. **Artifact Organization**: Structured storage of all experiment outputs
5. **Error Handling**: Graceful fallbacks when MLflow is unavailable
6. **Documentation**: Inline comments and structured logging

## Integration with Existing Workflow

### DVC Pipeline Compatibility:
The enhanced scripts work seamlessly with the existing DVC pipeline:
```bash
dvc repro  # Runs the entire pipeline with MLflow tracking
```

### Model Deployment:
Models can be served directly from MLflow:
```bash
mlflow models serve -m "models:/diabetes_model/Production" -p 1234
```

## Monitoring and Analysis

### 1. **Performance Tracking**
- Compare model performance across different experiments
- Track metric improvements over time
- Identify best performing hyperparameter combinations

### 2. **Model Lineage**
- Track which data and parameters produced which models
- Reproduce specific model versions
- Understand model evolution

### 3. **Collaboration**
- Share experiment results with team members
- Centralized model registry for team access
- Standardized experiment tracking

## Troubleshooting

### Common Issues:

1. **MLflow UI not accessible**:
   - Check if port 5000 is available
   - Use `--host 0.0.0.0` for external access

2. **Model loading errors**:
   - Verify MLflow tracking URI
   - Check model registry for registered models
   - Fallback to local model files available

3. **Artifact storage issues**:
   - Ensure write permissions in project directory
   - Check available disk space
   - Verify artifact_location in config

### Recovery Commands:
```bash
# Reset MLflow experiments (careful: this deletes all data)
rm -rf mlruns/ mlartifacts/

# Re-initialize with fresh tracking
python train.py
```

## Next Steps

1. **Remote Tracking**: Configure remote MLflow tracking server
2. **Model Serving**: Set up MLflow model serving endpoints  
3. **CI/CD Integration**: Integrate with automated training pipelines
4. **A/B Testing**: Implement model comparison frameworks
5. **Monitoring**: Add model performance monitoring in production

This comprehensive MLflow integration provides a robust foundation for experiment tracking, model management, and deployment in the diabetes prediction project.