# 🏥 MLOps Diabetes Prediction Project - Implementation Guide

## 📋 Project Overview

This document explains the complete MLOps implementation for the Diabetes Prediction project, covering repository structure, CI/CD pipelines, local deployment, and containerization.

---

## F. Repository Structure & Version Control

### Project Directory Structure

```
Mtech-diabetes-mlops-project-main/
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml              # GitHub Actions CI/CD pipeline
│
├── data/
│   ├── diabetes.csv               # Original dataset
│   ├── diabetes.csv.dvc           # DVC tracking file
│   ├── diabetes_processed.csv     # Preprocessed data
│   ├── test.csv                   # Test dataset
│   └── new_data.csv               # New data for predictions
│
├── model/
│   └── diabetes_model.pkl         # Trained ML model
│
├── src/
│   ├── __init__.py
│   ├── load_data.py               # Data loading utilities
│   ├── mlflow_utils.py            # MLflow helper functions
│   ├── pipeline.py                # ML pipeline components
│   ├── predict.py                 # Prediction utilities
│   └── train.py                   # Training utilities
│
├── tests/
│   ├── __init__.py
│   ├── test_load_data.py          # Data loading tests
│   ├── test_predict.py            # Prediction tests
│   └── test_train.py              # Training tests
│
├── mlruns/                        # MLflow experiment tracking
│
├── venv/                          # Python virtual environment
│
├── api.py                         # FastAPI application
├── streamlit_app.py               # Streamlit dashboard
├── preprocess.py                  # Data preprocessing script
├── train.py                       # Model training script
├── train_simple.py                # Simplified training script
├── evaluate.py                    # Model evaluation script
├── run_local_cicd.py              # Local CI/CD simulation
│
├── Dockerfile                     # Docker image for API
├── Dockerfile.streamlit           # Docker image for Streamlit
├── docker-compose.yml             # Docker orchestration
├── .dockerignore                  # Docker build exclusions
│
├── requirements.txt               # Python dependencies
├── mlflow_config.yaml             # MLflow configuration
├── dvc.yaml                       # DVC pipeline configuration
└── README.md                      # Project documentation
```

### Version Control Setup

1. **Git** - Source code versioning
2. **DVC (Data Version Control)** - Dataset and model versioning
3. **MLflow** - Experiment tracking and model registry

### Key Files Created/Modified

| File | Purpose |
|------|---------|
| `.github/workflows/ci-cd.yml` | GitHub Actions CI/CD pipeline |
| `run_local_cicd.py` | Local CI/CD simulation script |
| `Dockerfile` | Multi-stage Docker build for API |
| `Dockerfile.streamlit` | Docker build for dashboard |
| `docker-compose.yml` | Container orchestration |
| `streamlit_app.py` | Interactive prediction dashboard |
| `.dockerignore` | Optimizes Docker build context |

---

## G. CI/CD using GitHub Actions (Local Simulation)

### GitHub Actions Pipeline (`.github/workflows/ci-cd.yml`)

The CI/CD pipeline consists of 6 jobs:

#### 1. **Lint** - Code Quality Check
```yaml
- Flake8 for syntax errors and code quality
- Black for code formatting
- isort for import sorting
```

#### 2. **Test** - Unit Tests
```yaml
- pytest for running unit tests
- Coverage reporting
- Test artifacts upload
```

#### 3. **Build Model** - Training Pipeline
```yaml
- Data preprocessing
- Model training
- Model evaluation
- Artifact storage
```

#### 4. **API Test** - Integration Tests
```yaml
- Start API server
- Test health endpoint
- Test documentation endpoint
```

#### 5. **Docker Build** - Container Testing
```yaml
- Build Docker image
- Run container
- Verify health check
```

#### 6. **Deploy** - Deployment Simulation
```yaml
- Only runs on main branch
- Simulates production deployment
- Creates deployment summary
```

### Local CI/CD Simulation

Run the local CI/CD pipeline:

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run local CI/CD simulation
python run_local_cicd.py
```

**Pipeline Stages:**
1. ✅ Environment Check - Python & pip verification
2. ✅ Install Dependencies - Testing packages
3. ✅ Code Quality Checks - Flake8, Black, isort
4. ✅ Unit Tests - pytest with 7 tests
5. ✅ Data Pipeline - Preprocessing
6. ✅ Model Training - train_simple.py
7. ✅ Model Evaluation - evaluate.py
8. ✅ Docker Check - Build verification

**Expected Output:**
```
✅ CI/CD Pipeline PASSED
Total stages: 12
Passed: 12
Failed: 0
```

---

## H. Local Model Deployment via FastAPI/Streamlit

### FastAPI Deployment

#### Starting the API Server

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Start FastAPI server
uvicorn api:app --host 0.0.0.0 --port 8000
```

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check & model status |
| `/predict` | POST | Single patient prediction |
| `/predict/batch` | POST | Batch predictions (up to 1000) |
| `/predict/file` | POST | CSV file upload predictions |
| `/model/info` | GET | Model information |
| `/docs` | GET | Swagger UI documentation |
| `/redoc` | GET | ReDoc documentation |

#### Example API Request

```powershell
# Health Check
Invoke-RestMethod -Uri "http://localhost:8000/health"

# Single Prediction
$body = @{
    pregnancies = 2
    glucose = 120
    blood_pressure = 70
    skin_thickness = 20
    insulin = 80
    bmi = 25
    diabetes_pedigree_function = 0.5
    age = 30
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $body -ContentType "application/json"
```

#### Example Response

```json
{
    "prediction": 0,
    "probability": 0.14,
    "confidence": 0.86,
    "risk_level": "Low"
}
```

### Streamlit Dashboard

#### Starting the Dashboard

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Start Streamlit
streamlit run streamlit_app.py
```

**Access:** http://localhost:8501

#### Dashboard Features

1. **🔮 Make Prediction** - Single patient prediction with interactive form
2. **📊 Batch Prediction** - CSV file upload for multiple patients
3. **📈 Model Info** - Display model details and metrics
4. **ℹ️ About** - Project information

#### Input Features

| Feature | Description | Range |
|---------|-------------|-------|
| Pregnancies | Number of pregnancies | 0-20 |
| Glucose | Plasma glucose (mg/dL) | 0-300 |
| Blood Pressure | Diastolic BP (mmHg) | 0-200 |
| Skin Thickness | Triceps fold (mm) | 0-100 |
| Insulin | 2-Hour serum insulin (μU/mL) | 0-1000 |
| BMI | Body Mass Index | 0-70 |
| Diabetes Pedigree Function | DPF score | 0-3 |
| Age | Patient age (years) | 1-120 |

---

## I. Containerization using Docker

### Docker Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Compose                        │
├─────────────────┬─────────────────┬─────────────────────┤
│   API Service   │  MLflow Server  │ Streamlit (optional)│
│   Port: 8000    │   Port: 5000    │    Port: 8501       │
├─────────────────┴─────────────────┴─────────────────────┤
│                   mlops-network                          │
└─────────────────────────────────────────────────────────┘
```

### Dockerfile (Multi-stage Build)

```dockerfile
# Stage 1: Builder - Install dependencies
FROM python:3.10-slim as builder
# ... installs all Python packages

# Stage 2: Production - Optimized runtime
FROM python:3.10-slim as production
# ... copies only necessary files
# ... runs as non-root user for security
```

**Key Features:**
- Multi-stage build for smaller image size
- Non-root user for security
- Health check configuration
- Environment variables for configuration

### Docker Commands

#### Build and Run Individual Container

```powershell
# Build the image
docker build -t diabetes-mlops:latest .

# Run the container
docker run -d -p 8000:8000 --name diabetes-api diabetes-mlops:latest

# Check logs
docker logs diabetes-api

# Stop and remove
docker stop diabetes-api && docker rm diabetes-api
```

#### Docker Compose (Full Stack)

```powershell
# Start all services
docker-compose up -d

# Start with Streamlit dashboard
docker-compose --profile dashboard up -d

# View running containers
docker-compose ps

# View logs
docker-compose logs -f

# Stop all services
docker-compose down

# Rebuild after code changes
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Services Overview

| Service | Image | Port | Description |
|---------|-------|------|-------------|
| `api` | diabetes-mlops-api:latest | 8000 | FastAPI prediction service |
| `mlflow` | python:3.10-slim | 5000 | MLflow tracking server |
| `streamlit` | diabetes-mlops-streamlit:latest | 8501 | Interactive dashboard |

### Accessing Services

- **API Documentation:** http://localhost:8000/docs
- **API Health Check:** http://localhost:8000/health
- **MLflow UI:** http://localhost:5000
- **Streamlit Dashboard:** http://localhost:8501 (with `--profile dashboard`)

---

## 🚀 Quick Start Guide

### 1. Setup Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Model

```powershell
python train_simple.py
```

### 3. Run Local Services

```powershell
# Terminal 1: FastAPI
uvicorn api:app --host 0.0.0.0 --port 8000

# Terminal 2: Streamlit
streamlit run streamlit_app.py
```

### 4. Run CI/CD Simulation

```powershell
python run_local_cicd.py
```

### 5. Docker Deployment

```powershell
docker-compose up -d
```

---

## 📊 Test Results Summary

### Unit Tests (7 tests)

| Test | Status |
|------|--------|
| `test_load_data.py::test_load_data` | ✅ PASSED |
| `test_predict.py::test_model_file` | ✅ PASSED |
| `test_predict.py::test_model_prediction` | ✅ PASSED |
| `test_predict.py::test_model_probability` | ✅ PASSED |
| `test_train.py::test_model_exists` | ✅ PASSED |
| `test_train.py::test_model_can_load` | ✅ PASSED |
| `test_train.py::test_model_has_predict_proba` | ✅ PASSED |

### CI/CD Pipeline (12 stages)

| Stage | Status |
|-------|--------|
| Environment Check | ✅ PASS |
| Pip Check | ✅ PASS |
| Dependencies | ✅ PASS |
| Flake8 Critical | ✅ PASS |
| Flake8 Style | ✅ PASS |
| Black | ✅ PASS |
| isort | ✅ PASS |
| Unit Tests | ✅ PASS |
| Preprocessing | ✅ PASS |
| Training | ✅ PASS |
| Evaluation | ✅ PASS |
| Docker Build | ✅ PASS |

---

## 🔧 Configuration Files

### requirements.txt

```
pandas==2.0.3
scikit-learn==1.3.0
numpy==1.24.3
mlflow==2.8.1
joblib==1.3.2
dvc==3.30.3
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
python-multipart==0.0.6
matplotlib>=3.7.0
seaborn>=0.12.0
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-asyncio>=0.21.0
httpx>=0.25.0
flake8>=6.1.0
black>=23.9.0
isort>=5.12.0
streamlit>=1.28.0
requests>=2.31.0
```

### mlflow_config.yaml

```yaml
experiment_name: diabetes_prediction
tracking_uri: file:./mlruns
artifact_location: ./mlartifacts
registered_model_name: diabetes_model
run_name_prefix: diabetes_run
```

---

## 📝 Notes

1. **Model Accuracy:** ~74.68% on test data
2. **API Response Time:** < 100ms for single predictions
3. **Docker Image Size:** Optimized with multi-stage build
4. **Security:** Non-root user in Docker, input validation in API

---

## 🎯 Summary

This implementation provides a complete MLOps pipeline with:

- ✅ **Version Control** - Git + DVC + MLflow
- ✅ **CI/CD Pipeline** - GitHub Actions + Local simulation
- ✅ **API Deployment** - FastAPI with automatic documentation
- ✅ **Dashboard** - Streamlit for interactive predictions
- ✅ **Containerization** - Docker + Docker Compose
- ✅ **Testing** - pytest with 7 unit tests
- ✅ **Code Quality** - flake8, black, isort

---

*Generated on: December 6, 2025*
*Project: MTech Diabetes MLOps Project*
