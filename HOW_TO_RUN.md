# How to Run and Use the Diabetes Prediction MLOps Project

This guide provides step-by-step instructions to set up, run, and use all components of this MLOps project.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Setup Options](#setup-options)
   - [Option A: Local Development](#option-a-local-development)
   - [Option B: Docker Deployment](#option-b-docker-deployment)
4. [Using the API](#using-the-api)
5. [Using the Streamlit Dashboard](#using-the-streamlit-dashboard)
6. [Running Tests & CI/CD](#running-tests--cicd)
7. [MLflow Experiment Tracking](#mlflow-experiment-tracking)
8. [Training a New Model](#training-a-new-model)
9. [Local Monitoring with Evidently](#local-monitoring-with-evidently)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

| Software | Version | Download |
|----------|---------|----------|
| Python | 3.10+ | [python.org](https://www.python.org/downloads/) |
| Git | Latest | [git-scm.com](https://git-scm.com/downloads) |
| Docker | Latest | [docker.com](https://www.docker.com/products/docker-desktop/) |

### Hardware Requirements
- **RAM**: Minimum 4GB (8GB recommended)
- **Disk**: At least 2GB free space
- **Ports**: 8000, 5000, 8501 available

---

## Quick Start

### Option 1: Docker Hub (Fastest - No Build Required)

```powershell
# Create network and run pre-built images
docker network create diabetes-net
docker run -d -p 8000:8000 --name api --network diabetes-net aniketh703/diabetes-mlops-api:latest
docker run -d -p 8501:8501 --name streamlit --network diabetes-net -e API_URL=http://api:8000 aniketh703/diabetes-mlops-streamlit:latest
```

### Option 2: Docker Compose (Local Build)

```powershell
git clone <repository-url>
cd Mtech-diabetes-mlops-project-main
docker-compose --profile dashboard up -d
```

### Option 3: Local Development

```powershell
git clone <repository-url>
cd Mtech-diabetes-mlops-project-main
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

**Access Points:**

- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Streamlit: http://localhost:8501
- MLflow: http://localhost:5000

---

## Setup Options

### Option A: Local Development

#### Step 1: Create Virtual Environment

```powershell
# Navigate to project directory
cd C:\Users\Ani\OneDrive\Desktop\Mtech-diabetes-mlops-project-main

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# For Command Prompt use:
# venv\Scripts\activate.bat

# For Git Bash / Linux / Mac:
# source venv/bin/activate
```

#### Step 2: Install Dependencies

```powershell
# Install all required packages
pip install -r requirements.txt
```

#### Step 3: Run the API Server

```powershell
# Start FastAPI server
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

The `--reload` flag enables auto-reload on code changes (useful for development).

#### Step 4: Run MLflow Server (Optional)

```powershell
# In a new terminal, activate venv first
.\venv\Scripts\Activate.ps1

# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000
```

---

### Option B: Docker Deployment

#### Method 1: Using Docker Compose (Local Build)

```powershell
# Build and start all services in detached mode
docker-compose up -d --build

# Start with Streamlit dashboard included
docker-compose --profile dashboard up -d

# View running containers
docker-compose ps

# Check logs
docker-compose logs -f
```

#### Method 2: Using Docker Hub Images (Pre-built)

Images are available on Docker Hub at:
- `aniketh703/diabetes-mlops-api:latest`
- `aniketh703/diabetes-mlops-streamlit:latest`

```powershell
# Step 1: Create a shared network
docker network create diabetes-net

# Step 2: Run API container
docker run -d -p 8000:8000 --name api --network diabetes-net aniketh703/diabetes-mlops-api:latest

# Step 3: Run Streamlit container
docker run -d -p 8501:8501 --name streamlit --network diabetes-net -e API_URL=http://api:8000 aniketh703/diabetes-mlops-streamlit:latest
```

#### Method 3: Docker Hub with Local Data Volumes

Mount your local data and model folders for easy updates:

```powershell
# Create network
docker network create diabetes-net

# Run API with volume mounts
docker run -d -p 8000:8000 --name api --network diabetes-net `
  -v "${PWD}/data:/app/data" `
  -v "${PWD}/model:/app/model" `
  aniketh703/diabetes-mlops-api:latest

# Run Streamlit
docker run -d -p 8501:8501 --name streamlit --network diabetes-net `
  -e API_URL=http://api:8000 `
  aniketh703/diabetes-mlops-streamlit:latest
```

#### Updating Data in Docker

When using volume mounts, update data locally:

```powershell
# Update training data
Copy-Item new_data.csv data/diabetes.csv

# Retrain model
python train_simple.py

# Restart API to reload model
docker restart api
```

#### Verify Services

```powershell
# Check API health
curl http://localhost:8000/health

# Or using PowerShell
Invoke-RestMethod -Uri "http://localhost:8000/health"

# Check running containers
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

#### Stop Services

```powershell
# Stop individual containers
docker stop api streamlit
docker rm api streamlit

# Or stop all containers
docker rm -f $(docker ps -aq)

# Remove network
docker network rm diabetes-net

# If using docker-compose
docker-compose down
docker-compose down -v  # Also removes volumes
```

---

## Using the API

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message |
| `/health` | GET | Health check with model status |
| `/predict` | POST | Make a diabetes prediction |
| `/docs` | GET | Interactive API documentation |
| `/redoc` | GET | Alternative API documentation |

### Making Predictions

#### Using PowerShell

```powershell
# Single prediction
$body = @{
    pregnancies = 6
    glucose = 148
    blood_pressure = 72
    skin_thickness = 35
    insulin = 0
    bmi = 33.6
    diabetes_pedigree_function = 0.627
    age = 50
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $body -ContentType "application/json"
```

#### Using cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pregnancies": 6,
    "glucose": 148,
    "blood_pressure": 72,
    "skin_thickness": 35,
    "insulin": 0,
    "bmi": 33.6,
    "diabetes_pedigree_function": 0.627,
    "age": 50
  }'
```

#### Using Python

```python
import requests

data = {
    "pregnancies": 6,
    "glucose": 148,
    "blood_pressure": 72,
    "skin_thickness": 35,
    "insulin": 0,
    "bmi": 33.6,
    "diabetes_pedigree_function": 0.627,
    "age": 50
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

#### Using the API Test Script

```powershell
# Run the built-in API client
python api_client.py
```

### Understanding the Response

```json
{
    "prediction": 1,
    "probability": 0.73,
    "risk_level": "High Risk",
    "message": "Based on the input features, the model predicts diabetes risk."
}
```

| Field | Description |
|-------|-------------|
| `prediction` | 0 = No Diabetes, 1 = Diabetes |
| `probability` | Confidence score (0.0 to 1.0) |
| `risk_level` | Low/Medium/High Risk classification |
| `message` | Descriptive result message |

### Input Features Reference

| Feature | Description | Typical Range |
|---------|-------------|---------------|
| `pregnancies` | Number of pregnancies | 0-17 |
| `glucose` | Plasma glucose concentration | 0-200 |
| `blood_pressure` | Diastolic blood pressure (mm Hg) | 0-122 |
| `skin_thickness` | Triceps skin fold thickness (mm) | 0-99 |
| `insulin` | 2-Hour serum insulin (mu U/ml) | 0-846 |
| `bmi` | Body mass index | 0-67.1 |
| `diabetes_pedigree_function` | Diabetes pedigree function | 0.078-2.42 |
| `age` | Age in years | 21-81 |

---

## Using the Streamlit Dashboard

### Start Streamlit

```powershell
# Make sure venv is activated
.\venv\Scripts\Activate.ps1

# Run Streamlit app
streamlit run streamlit_app.py
```

Access at: http://localhost:8501

### Features

1. **Single Prediction**: Use sliders to input patient data and get instant predictions
2. **Batch Prediction**: Upload a CSV file for multiple predictions
3. **Visualization**: View probability gauges and risk assessments

### Batch Prediction CSV Format

Create a CSV file with these columns:

```csv
pregnancies,glucose,blood_pressure,skin_thickness,insulin,bmi,diabetes_pedigree_function,age
6,148,72,35,0,33.6,0.627,50
1,85,66,29,0,26.6,0.351,31
```

---

## Running Tests & CI/CD

### Run Unit Tests

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Run Local CI/CD Pipeline

```powershell
# Run the full CI/CD simulation locally
python run_local_cicd.py
```

This runs 12 stages:
1. ✅ Environment Setup
2. ✅ Dependency Installation
3. ✅ Code Formatting Check (Black)
4. ✅ Import Sorting Check (isort)
5. ✅ Linting (flake8)
6. ✅ Unit Tests
7. ✅ Data Validation
8. ✅ Model Validation
9. ✅ API Health Check
10. ✅ Integration Tests
11. ✅ Security Check
12. ✅ Build Verification

### Run Code Quality Checks Manually

```powershell
# Format code with Black
black . --check

# Sort imports with isort
isort . --check-only

# Lint with flake8
flake8 . --max-line-length=120 --exclude=venv,__pycache__,.git
```

---

## MLflow Experiment Tracking

### Access MLflow UI

```powershell
# Start MLflow server
mlflow ui --host 0.0.0.0 --port 5000
```

Access at: http://localhost:5000

### View Experiments

1. Open MLflow UI in browser
2. Click on "Diabetes_Prediction" experiment
3. View runs, metrics, parameters, and artifacts

### Log a New Experiment Run

```python
import mlflow

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Diabetes_Prediction")

with mlflow.start_run():
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("accuracy", 0.85)
    mlflow.sklearn.log_model(model, "model")
```

---

## Training a New Model

### Using the Training Script

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run training (simple version)
python train_simple.py

# Run training (with MLflow tracking)
python train.py
```

### Training Pipeline with DVC

```powershell
# Run the full DVC pipeline
dvc repro

# View pipeline status
dvc status
```

### Custom Training

```python
from src.train import train_model
from src.load_data import load_data

# Load data
X_train, X_test, y_train, y_test = load_data("data/diabetes.csv")

# Train model
model = train_model(X_train, y_train)

# Save model
import joblib
joblib.dump(model, "model/diabetes_model.pkl")
```

---

## Local Monitoring with Evidently

Evidently is an open-source tool for monitoring ML model performance and data drift. Use it locally to track how your model performs over time.

### Install Evidently

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install Evidently (compatible version)
pip install evidently==0.4.33
```

### Run Monitoring

The project includes a pre-built monitoring script `monitor_model.py`:

```powershell
# Run the monitoring script
python monitor_model.py

# Open the reports in browser
Start-Process "reports/data_drift_report.html"
Start-Process "reports/data_quality_report.html"
```

### What the Script Does

The `monitor_model.py` script:
1. Loads reference data (training dataset: `data/diabetes.csv`)
2. Loads current/production data (`data/new_data.csv`)
3. Compares feature distributions between datasets
4. Generates HTML reports in the `reports/` folder

### Generated Reports

| Report | Location | Description |
|--------|----------|-------------|
| Data Drift | `reports/data_drift_report.html` | Shows if input features have shifted |
| Data Quality | `reports/data_quality_report.html` | Shows missing values, anomalies |

### What Evidently Monitors

| Report Type | Description |
|-------------|-------------|
| **Data Drift** | Detects if input feature distributions have changed |
| **Data Quality** | Checks for missing values, duplicates, anomalies |
| **Target Drift** | Monitors if target variable distribution shifts |
| **Classification Performance** | Tracks accuracy, precision, recall, F1 over time |

### Quick Monitoring Commands

```powershell
# === EVIDENTLY MONITORING ===
pip install evidently==0.4.33                    # Install Evidently
python monitor_model.py                          # Generate reports
Start-Process reports/data_drift_report.html    # View drift report
Start-Process reports/data_quality_report.html  # View quality report
```

---

## Troubleshooting

### Common Issues

#### 1. Port Already in Use

```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F
```

#### 2. Virtual Environment Not Activating

```powershell
# Check execution policy
Get-ExecutionPolicy

# If restricted, run PowerShell as Admin and execute:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### 3. Docker Container Not Starting

```powershell
# Check container logs
docker-compose logs api

# Rebuild containers
docker-compose down
docker-compose up -d --build
```

#### 4. Model Not Found Error

```powershell
# Ensure model exists
Test-Path "model/diabetes_model.pkl"

# If not, train a new model
python train_simple.py
```

#### 5. Import Errors

```powershell
# Ensure you're in the project root directory
cd C:\Users\Ani\OneDrive\Desktop\Mtech-diabetes-mlops-project-main

# Reinstall dependencies
pip install -r requirements.txt
```

#### 6. MLflow Errors

```powershell
# Clear MLflow cache
Remove-Item -Recurse -Force mlruns\.trash -ErrorAction SilentlyContinue

# Restart MLflow
mlflow ui --host 0.0.0.0 --port 5000
```

### Getting Help

1. Check the `FastAPI_Documentation.md` for API details
2. Check the `MLflow_Integration_Guide.md` for MLflow setup
3. Check the `PROJECT_IMPLEMENTATION.md` for implementation details
4. Review test files in `tests/` for usage examples

---

## Quick Reference Commands

```powershell
# === SETUP ===
python -m venv venv                              # Create venv
.\venv\Scripts\Activate.ps1                      # Activate venv
pip install -r requirements.txt                  # Install deps

# === RUN SERVICES ===
uvicorn api:app --host 0.0.0.0 --port 8000      # Start API
streamlit run streamlit_app.py                   # Start Streamlit
mlflow ui --host 0.0.0.0 --port 5000            # Start MLflow

# === DOCKER ===
docker-compose up -d                             # Start containers
docker-compose down                              # Stop containers
docker-compose logs -f                           # View logs

# === TESTING ===
pytest tests/ -v                                 # Run tests
python run_local_cicd.py                         # Run CI/CD

# === TRAINING ===
python train_simple.py                           # Train model
dvc repro                                        # Run DVC pipeline
```

---

## Service URLs Summary

| Service | URL | Description |
|---------|-----|-------------|
| FastAPI | http://localhost:8000 | Main API |
| Swagger Docs | http://localhost:8000/docs | Interactive API docs |
| ReDoc | http://localhost:8000/redoc | Alternative API docs |
| Health Check | http://localhost:8000/health | API health status |
| MLflow | http://localhost:5000 | Experiment tracking |
| Streamlit | http://localhost:8501 | Interactive dashboard |

---

**Happy Predicting! 🩺📊**
