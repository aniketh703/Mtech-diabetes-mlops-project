# 🏥 MLOps Diabetes Prediction Project

A comprehensive MLOps pipeline for diabetes prediction using machine learning with MLflow experiment tracking, DVC data versioning, and FastAPI deployment.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [MLOps Pipeline](#mlops-pipeline)
- [Data Management](#data-management)
- [Model Training](#model-training)
- [Testing](#testing)
- [Contributing](#contributing)

## 🎯 Overview

This project implements an end-to-end MLOps pipeline for diabetes prediction using the Pima Indians Diabetes Database. The system includes data preprocessing, model training, experiment tracking, model evaluation, and deployment through a RESTful API.

### Key Technologies
- **Machine Learning**: scikit-learn, pandas, numpy
- **MLOps**: MLflow for experiment tracking and model management
- **Data Versioning**: DVC (Data Version Control)
- **API Framework**: FastAPI with automatic documentation
- **Model Deployment**: Uvicorn ASGI server
- **Testing**: pytest for unit testing

## ✨ Features

### 🤖 Machine Learning Pipeline
- ✅ Data preprocessing and feature engineering
- ✅ Logistic regression model training
- ✅ Model evaluation with comprehensive metrics
- ✅ Hyperparameter tracking
- ✅ Model versioning and artifacts storage

### 🔄 MLOps Capabilities
- ✅ Experiment tracking with MLflow
- ✅ Data versioning with DVC
- ✅ Automated pipeline orchestration
- ✅ Model performance monitoring
- ✅ Reproducible experiments

### 🚀 API Features
- ✅ Single patient prediction
- ✅ Batch predictions (up to 1000 samples)
- ✅ File-based predictions (CSV upload)
- ✅ Model health checks and information
- ✅ Comprehensive input validation
- ✅ Background model retraining
- ✅ Risk level classification

## 📁 Project Structure

```
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── dvc.yaml                    # DVC pipeline configuration
├── mlflow_config.yaml          # MLflow configuration
├── start_api.sh               # API startup script
│
├── data/                      # Data directory
│   ├── diabetes.csv           # Original dataset
│   ├── diabetes.csv.dvc       # DVC tracking file
│   ├── diabetes_processed.csv # Preprocessed data
│   ├── test.csv              # Test dataset
│   └── new_data.csv          # New data for predictions
│
├── model/                     # Model artifacts
│   └── diabetes_model.pkl     # Trained model
│
├── src/                       # Source code modules
│   ├── __init__.py
│   ├── load_data.py          # Data loading utilities
│   ├── mlflow_utils.py       # MLflow helper functions
│   ├── pipeline.py           # ML pipeline components
│   ├── predict.py            # Prediction utilities
│   └── train.py              # Training utilities
│
├── tests/                     # Unit tests
│   ├── __init__.py
│   ├── test_load_data.py
│   ├── test_predict.py
│   └── test_train.py
│
├── mlruns/                    # MLflow experiment tracking
│   └── [experiment_folders]
│
├── api.py                     # FastAPI application
├── api_client.py              # API client example
├── api_test_data.csv          # API testing data
├── preprocess.py              # Data preprocessing script
├── train.py                   # Model training script
├── train_simple.py            # Simplified training script
├── evaluate.py                # Model evaluation script
├── mlflow_manager.py          # MLflow management utilities
├── FastAPI_Documentation.md   # Detailed API documentation
└── MLflow_Integration_Guide.md # MLflow setup guide
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Mtech-diabetes-mlops-project-main
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize DVC** (if not already done)
   ```bash
   dvc init
   dvc pull  # Pull data if available in remote storage
   ```

5. **Set up MLflow**
   ```bash
   # MLflow tracking server will start automatically with the API
   # Or manually start: mlflow ui --host 0.0.0.0 --port 5000
   ```

## 🚀 Usage

### Quick Start

1. **Run the complete DVC pipeline**
   ```bash
   dvc repro
   ```
   This will execute: data preprocessing → model training → model evaluation

2. **Start the FastAPI server**
   ```bash
   # Using the startup script
   ./start_api.sh
   
   # Or manually
   uvicorn api:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Access the API**
   - API Documentation: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc
   - Health check: http://localhost:8000/health

### Manual Pipeline Execution

1. **Data Preprocessing**
   ```bash
   python preprocess.py
   ```

2. **Model Training**
   ```bash
   python train.py
   ```

3. **Model Evaluation**
   ```bash
   python evaluate.py
   ```

## 📚 API Documentation

### Core Endpoints

#### Health Check
```http
GET /health
```
Returns API health status and model information.

#### Single Prediction
```http
POST /predict
Content-Type: application/json

{
    "pregnancies": 2,
    "glucose": 120,
    "blood_pressure": 70,
    "skin_thickness": 25,
    "insulin": 80,
    "bmi": 25.5,
    "diabetes_pedigree_function": 0.5,
    "age": 30
}
```

#### Batch Prediction
```http
POST /predict/batch
Content-Type: application/json

{
    "patients": [
        {
            "pregnancies": 2,
            "glucose": 120,
            // ... other features
        },
        // ... more patients (up to 1000)
    ]
}
```

#### File Upload Prediction
```http
POST /predict/file
Content-Type: multipart/form-data

file: [CSV file with patient data]
```

### Response Format

```json
{
    "prediction": 1,
    "probability": 0.75,
    "risk_level": "High",
    "confidence": "High",
    "model_version": "1.0.0",
    "timestamp": "2025-12-06T10:00:00Z"
}
```

For detailed API documentation, see [FastAPI_Documentation.md](FastAPI_Documentation.md).

## 🔄 MLOps Pipeline

### DVC Pipeline Stages

1. **Preprocessing** (`preprocess.py`)
   - Data cleaning and validation
   - Feature engineering
   - Data splitting

2. **Training** (`train.py`)
   - Model training with hyperparameter logging
   - MLflow experiment tracking
   - Model artifact storage

3. **Evaluation** (`evaluate.py`)
   - Model performance assessment
   - Metrics calculation and logging
   - Visualization generation

### MLflow Integration

- **Experiment Tracking**: All runs logged with parameters, metrics, and artifacts
- **Model Registry**: Best models registered for deployment
- **Artifact Storage**: Model files and plots stored automatically
- **Comparison**: Easy comparison of different runs and parameters

For MLflow setup details, see [MLflow_Integration_Guide.md](MLflow_Integration_Guide.md).

## 💾 Data Management

### Dataset Information
- **Source**: Pima Indians Diabetes Database
- **Features**: 8 medical diagnostic measurements
- **Target**: Binary diabetes outcome (0: No, 1: Yes)
- **Size**: ~768 patients

### Features Description
- `pregnancies`: Number of times pregnant
- `glucose`: Plasma glucose concentration (mg/dL)
- `blood_pressure`: Diastolic blood pressure (mm Hg)
- `skin_thickness`: Triceps skin fold thickness (mm)
- `insulin`: 2-Hour serum insulin (mu U/ml)
- `bmi`: Body mass index (weight in kg/(height in m)^2)
- `diabetes_pedigree_function`: Diabetes pedigree function
- `age`: Age (years)

### DVC Data Versioning
```bash
# Track data changes
dvc add data/diabetes.csv

# Commit changes
git add data/diabetes.csv.dvc .gitignore
git commit -m "Add dataset"

# Push to remote storage
dvc push
```

## 🎯 Model Training

### Training Process

1. **Data Loading**: Load and validate dataset
2. **Preprocessing**: Handle missing values, scaling, feature engineering
3. **Model Training**: Logistic Regression with cross-validation
4. **Evaluation**: Calculate metrics (accuracy, precision, recall, F1-score)
5. **Logging**: Record everything in MLflow

### Model Performance Metrics
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve

### Hyperparameter Tuning
```python
# Example hyperparameters tracked
{
    "C": 1.0,
    "max_iter": 100,
    "solver": "liblinear",
    "test_size": 0.2,
    "random_state": 42
}
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_predict.py

# Run with coverage
pytest --cov=src tests/
```

### Test Coverage
- **Data Loading**: Validate data loading and preprocessing
- **Model Training**: Test training pipeline components
- **Predictions**: Verify prediction accuracy and format
- **API Endpoints**: Test all FastAPI endpoints

## 📊 Monitoring and Logging

### MLflow UI
Access MLflow experiments at: http://localhost:5000

### API Logs
- Request/response logging
- Error tracking
- Performance monitoring
- Model prediction logging

## 🔧 Configuration

### MLflow Configuration (`mlflow_config.yaml`)
```yaml
tracking_uri: "./mlruns"
experiment_name: "diabetes_prediction"
run_name_prefix: "diabetes_model"
```

### DVC Configuration (`dvc.yaml`)
- Pipeline stages definition
- Dependencies and outputs
- Metrics and plots configuration

## 🐳 Docker Deployment

### Build and Run with Docker

1. **Build the Docker image**
   ```bash
   docker build -t diabetes-mlops:latest .
   ```

2. **Run the container**
   ```bash
   docker run -d -p 8000:8000 --name diabetes-api diabetes-mlops:latest
   ```

3. **Access the API**
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs

### Docker Compose (Full Stack)

Run the complete stack including MLflow tracking server:

```bash
# Start all services
docker-compose up -d

# Start with Streamlit dashboard
docker-compose --profile dashboard up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

**Services Available:**
- **API**: http://localhost:8000 - FastAPI prediction service
- **MLflow**: http://localhost:5000 - Experiment tracking UI
- **Streamlit**: http://localhost:8501 - Interactive dashboard (with `--profile dashboard`)

## 🔄 CI/CD Pipeline

### GitHub Actions

The project includes automated CI/CD pipelines in `.github/workflows/ci-cd.yml`:

**Pipeline Stages:**
1. **Lint** - Code quality checks with flake8, black, and isort
2. **Test** - Unit tests with pytest and coverage reporting
3. **Build Model** - Train and evaluate the ML model
4. **API Test** - Integration tests for FastAPI endpoints
5. **Docker Build** - Build and test Docker container
6. **Deploy** - Deployment simulation (on main branch)

### Local CI/CD Simulation

Run the CI/CD pipeline locally:

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate     # Linux/Mac

# Run local CI/CD simulation
python run_local_cicd.py
```

### Manual Steps

```bash
# Linting
flake8 . --exclude=venv,__pycache__

# Format check
black --check .

# Run tests
pytest tests/ -v --cov=src

# Build Docker
docker build -t diabetes-mlops:test .
```

## 🖥️ Streamlit Dashboard

An interactive dashboard is available for making predictions:

```bash
# Run Streamlit locally
streamlit run streamlit_app.py

# Or with Docker
docker-compose --profile dashboard up -d
```

Access at: http://localhost:8501

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Write comprehensive tests
- Document new features
- Use meaningful commit messages
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For questions and support:
- Create an issue in the repository
- Check existing documentation
- Review MLflow and DVC official documentation

## 🎉 Acknowledgments

- Pima Indians Diabetes Database contributors
- MLflow development team
- DVC development team
- FastAPI framework developers
- scikit-learn contributors

---

**Built with ❤️ for MLOps and Healthcare AI**