"""
FastAPI application for Diabetes Prediction API

This API provides endpoints for:
1. Health checks
2. Model predictions (single and batch)
3. Model information and metrics
4. Data preprocessing
5. MLflow experiment tracking
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import os
import json
import io
from datetime import datetime
import logging

# Import project modules
from src.mlflow_utils import get_mlflow_manager
from src.load_data import load_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Diabetes Prediction API",
    description="MLOps API for diabetes prediction with MLflow integration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global variables for model and MLflow manager
model = None
mlflow_manager = None
model_info = {}

def load_model_on_startup():
    """Load the best model on application startup"""
    global model, mlflow_manager, model_info
    
    # First try to load from local file (more reliable)
    model_paths = ["model/diabetes_model.pkl", "artifacts/diabetes_model.pkl"]
    for path in model_paths:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                model_info = {
                    "source": "local_file",
                    "path": path,
                    "loaded_at": datetime.now().isoformat()
                }
                logger.info(f"Loaded model from local file: {path}")
                break
            except Exception as e:
                logger.warning(f"Failed to load model from {path}: {e}")
    
    # Initialize MLflow manager (optional, for tracking)
    try:
        mlflow_manager = get_mlflow_manager()
        logger.info("MLflow manager initialized")
    except Exception as e:
        logger.warning(f"MLflow manager initialization failed: {e}")
        mlflow_manager = None
    
    # If no local model, try MLflow
    if model is None and mlflow_manager is not None:
        try:
            best_run = mlflow_manager.get_best_run(metric_name="test_accuracy")
            if best_run is not None:
                run_id = best_run['run_id'] if isinstance(best_run, dict) else best_run.get('run_id', None)
                if run_id:
                    model = mlflow_manager.load_model_from_run(run_id)
                    model_info = {
                        "source": "mlflow",
                        "run_id": run_id,
                        "loaded_at": datetime.now().isoformat()
                    }
                    logger.info(f"Loaded model from MLflow run: {run_id}")
        except Exception as e:
            logger.warning(f"Failed to load model from MLflow: {e}")
    
    if model is None:
        logger.error("No model found!")

# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_model_on_startup()

# Pydantic models for request/response validation
class DiabetesInput(BaseModel):
    """Single diabetes prediction input"""
    pregnancies: int = Field(..., ge=0, le=20, description="Number of pregnancies")
    glucose: float = Field(..., gt=0, le=300, description="Glucose level (mg/dL)")
    blood_pressure: float = Field(..., gt=0, le=200, description="Blood pressure (mmHg)")
    skin_thickness: float = Field(..., ge=0, le=100, description="Skin thickness (mm)")
    insulin: float = Field(..., ge=0, le=1000, description="Insulin level (μU/mL)")
    bmi: float = Field(..., gt=0, le=70, description="Body Mass Index")
    diabetes_pedigree_function: float = Field(..., gt=0, le=3, description="Diabetes pedigree function")
    age: int = Field(..., gt=0, le=120, description="Age in years")
    
    class Config:
        schema_extra = {
            "example": {
                "pregnancies": 2,
                "glucose": 120.0,
                "blood_pressure": 80.0,
                "skin_thickness": 25.0,
                "insulin": 100.0,
                "bmi": 25.5,
                "diabetes_pedigree_function": 0.5,
                "age": 35
            }
        }

class DiabetesBatchInput(BaseModel):
    """Batch diabetes prediction input"""
    samples: List[DiabetesInput] = Field(..., min_items=1, max_items=1000)

class PredictionResponse(BaseModel):
    """Single prediction response"""
    prediction: int = Field(..., description="Prediction: 0 (No Diabetes), 1 (Diabetes)")
    probability: float = Field(..., description="Probability of having diabetes")
    confidence: float = Field(..., description="Prediction confidence")
    risk_level: str = Field(..., description="Risk level: Low, Medium, High")

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    summary: Dict[str, Any]

class ModelInfo(BaseModel):
    """Model information response"""
    model_loaded: bool
    model_info: Dict[str, Any]
    feature_names: List[str]
    model_type: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    model_loaded: bool
    mlflow_connection: bool

# Utility functions
def calculate_risk_level(probability: float) -> str:
    """Calculate risk level based on probability"""
    if probability < 0.3:
        return "Low"
    elif probability < 0.7:
        return "Medium"
    else:
        return "High"

def preprocess_input(input_data: DiabetesInput) -> np.ndarray:
    """Preprocess input data for prediction"""
    features = [
        input_data.pregnancies,
        input_data.glucose,
        input_data.blood_pressure,
        input_data.skin_thickness,
        input_data.insulin,
        input_data.bmi,
        input_data.diabetes_pedigree_function,
        input_data.age
    ]
    return np.array(features).reshape(1, -1)

async def log_prediction_to_mlflow(input_data: DiabetesInput, prediction: int, probability: float):
    """Log prediction to MLflow (background task)"""
    try:
        if mlflow_manager:
            with mlflow_manager.start_run(run_name="api_prediction"):
                mlflow.log_param("api_prediction", True)
                mlflow.log_param("input_glucose", input_data.glucose)
                mlflow.log_param("input_bmi", input_data.bmi)
                mlflow.log_param("input_age", input_data.age)
                mlflow.log_metric("prediction", prediction)
                mlflow.log_metric("probability", probability)
    except Exception as e:
        logger.error(f"Error logging to MLflow: {e}")

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Diabetes Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    mlflow_connection = False
    try:
        if mlflow_manager:
            mlflow_manager.get_experiment_info()
            mlflow_connection = True
    except:
        pass
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None,
        mlflow_connection=mlflow_connection
    )

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    feature_names = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]
    
    return ModelInfo(
        model_loaded=True,
        model_info=model_info,
        feature_names=feature_names,
        model_type=type(model).__name__
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_diabetes(
    input_data: DiabetesInput,
    background_tasks: BackgroundTasks,
    log_to_mlflow: bool = True
):
    """Make a single diabetes prediction"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess input
        features = preprocess_input(input_data)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        probability = probabilities[1]  # Probability of diabetes
        confidence = max(probabilities)
        risk_level = calculate_risk_level(probability)
        
        # Log to MLflow in background
        if log_to_mlflow:
            background_tasks.add_task(
                log_prediction_to_mlflow, 
                input_data, 
                prediction, 
                probability
            )
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            confidence=float(confidence),
            risk_level=risk_level
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_diabetes_batch(batch_input: DiabetesBatchInput):
    """Make batch diabetes predictions"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = []
        total_positive = 0
        total_high_risk = 0
        
        for input_data in batch_input.samples:
            features = preprocess_input(input_data)
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            probability = probabilities[1]
            confidence = max(probabilities)
            risk_level = calculate_risk_level(probability)
            
            if prediction == 1:
                total_positive += 1
            if risk_level == "High":
                total_high_risk += 1
            
            predictions.append(PredictionResponse(
                prediction=int(prediction),
                probability=float(probability),
                confidence=float(confidence),
                risk_level=risk_level
            ))
        
        summary = {
            "total_samples": len(batch_input.samples),
            "positive_predictions": total_positive,
            "negative_predictions": len(batch_input.samples) - total_positive,
            "high_risk_samples": total_high_risk,
            "positive_rate": total_positive / len(batch_input.samples)
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/predict/file")
async def predict_from_file(file: UploadFile = File(...)):
    """Make predictions from uploaded CSV file"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Read uploaded file
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        # Validate columns
        required_columns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing columns: {missing_columns}"
            )
        
        # Make predictions
        features = df[required_columns].values
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        
        # Add predictions to dataframe
        df['prediction'] = predictions
        df['diabetes_probability'] = probabilities[:, 1]
        df['confidence'] = np.max(probabilities, axis=1)
        df['risk_level'] = [calculate_risk_level(prob) for prob in probabilities[:, 1]]
        
        # Save results
        result_filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(result_filename, index=False)
        
        # Return file
        return FileResponse(
            result_filename,
            media_type='text/csv',
            filename=result_filename
        )
        
    except Exception as e:
        logger.error(f"File prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"File prediction failed: {str(e)}")

@app.post("/model/retrain")
async def trigger_model_retraining(background_tasks: BackgroundTasks):
    """Trigger model retraining (background task)"""
    def retrain_model():
        try:
            logger.info("Starting model retraining...")
            # Import and run training
            from train import main as train_main
            train_main()
            
            # Reload model
            load_model_on_startup()
            logger.info("Model retraining completed")
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
    
    background_tasks.add_task(retrain_model)
    return {"message": "Model retraining started", "status": "background_task_started"}

@app.get("/mlflow/experiments")
async def list_mlflow_experiments():
    """List MLflow experiments"""
    if mlflow_manager is None:
        raise HTTPException(status_code=503, detail="MLflow not available")
    
    try:
        experiment_info = mlflow_manager.get_experiment_info()
        runs = mlflow_manager.list_runs(max_results=10)
        
        return {
            "experiment": experiment_info,
            "recent_runs": runs.to_dict('records') if not runs.empty else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MLflow error: {str(e)}")

@app.get("/mlflow/best-model")
async def get_best_mlflow_model():
    """Get information about the best model from MLflow"""
    if mlflow_manager is None:
        raise HTTPException(status_code=503, detail="MLflow not available")
    
    try:
        best_run = mlflow_manager.get_best_run(metric_name="test_accuracy")
        if best_run is not None:
            return {
                "run_id": best_run['run_id'],
                "accuracy": best_run.get('test_accuracy'),
                "f1_score": best_run.get('test_f1_score'),
                "start_time": best_run['start_time'],
                "status": best_run['status']
            }
        else:
            return {"message": "No runs found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MLflow error: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found", "available_endpoints": ["/docs", "/health", "/predict"]}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "message": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )