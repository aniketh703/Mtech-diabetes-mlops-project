#!/bin/bash

# FastAPI Server Startup Script for Diabetes Prediction API

echo "=== Diabetes Prediction FastAPI Server ==="
echo "Starting server..."

# Check if model exists, if not train one
if [ ! -f "model/diabetes_model.pkl" ] && [ ! -f "artifacts/diabetes_model.pkl" ]; then
    echo "No model found. Training model first..."
    python train.py
fi

# Check if MLflow config exists
if [ ! -f "mlflow_config.yaml" ]; then
    echo "Warning: MLflow config not found. Some features may not work."
fi

# Start FastAPI server
echo "Starting FastAPI server on http://localhost:8000"
echo "API Documentation available at: http://localhost:8000/docs"
echo "Press Ctrl+C to stop the server"

uvicorn api:app --host 0.0.0.0 --port 8000 --reload