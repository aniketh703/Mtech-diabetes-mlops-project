# FastAPI Diabetes Prediction API Documentation

## Overview

This FastAPI application provides a comprehensive RESTful API for diabetes prediction with MLflow integration. The API is designed to be consumed by downstream services and applications.

## 🚀 **API Features**

### **Core Functionality**
- ✅ Single patient prediction
- ✅ Batch predictions (up to 1000 samples)
- ✅ File-based predictions (CSV upload)
- ✅ Model information and health checks
- ✅ MLflow experiment integration
- ✅ Background model retraining
- ✅ Comprehensive input validation

### **Enterprise Features**
- 🔒 Input validation with Pydantic models
- 📊 Detailed prediction confidence and risk levels
- 📈 MLflow experiment tracking
- 🔄 Background tasks for logging
- 📁 File upload/download capabilities
- ⚡ High-performance async endpoints

## 🏃‍♂️ **Quick Start**

### **1. Start the API Server**
```bash
# Method 1: Using startup script
./start_api.sh

# Method 2: Direct uvicorn
python3 -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# Method 3: Background mode
nohup python3 -m uvicorn api:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &
```

### **2. Access API Documentation**
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 📡 **API Endpoints**

### **Health & Information**

#### `GET /health`
**Purpose**: Check API health status
**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-12-06T21:05:47.869329",
  "model_loaded": true,
  "mlflow_connection": true
}
```

#### `GET /model/info`
**Purpose**: Get loaded model information
**Response**:
```json
{
  "model_loaded": true,
  "model_info": {
    "source": "local_file",
    "path": "model/diabetes_model.pkl",
    "loaded_at": "2025-12-06T21:05:06.020191"
  },
  "feature_names": ["Pregnancies", "Glucose", "BloodPressure", ...],
  "model_type": "LogisticRegression"
}
```

### **Prediction Endpoints**

#### `POST /predict`
**Purpose**: Single patient prediction
**Request Body**:
```json
{
  "pregnancies": 2,
  "glucose": 120.0,
  "blood_pressure": 80.0,
  "skin_thickness": 25.0,
  "insulin": 100.0,
  "bmi": 25.5,
  "diabetes_pedigree_function": 0.5,
  "age": 35
}
```

**Response**:
```json
{
  "prediction": 0,
  "probability": 0.1459368040996782,
  "confidence": 0.8540631959003218,
  "risk_level": "Low"
}
```

#### `POST /predict/batch`
**Purpose**: Batch predictions (up to 1000 samples)
**Request Body**:
```json
{
  "samples": [
    {
      "pregnancies": 2,
      "glucose": 120.0,
      "blood_pressure": 80.0,
      "skin_thickness": 25.0,
      "insulin": 100.0,
      "bmi": 25.5,
      "diabetes_pedigree_function": 0.5,
      "age": 35
    },
    { ... more samples ... }
  ]
}
```

**Response**:
```json
{
  "predictions": [
    {
      "prediction": 0,
      "probability": 0.146,
      "confidence": 0.854,
      "risk_level": "Low"
    }
  ],
  "summary": {
    "total_samples": 3,
    "positive_predictions": 1,
    "negative_predictions": 2,
    "high_risk_samples": 0,
    "positive_rate": 0.333
  }
}
```

#### `POST /predict/file`
**Purpose**: Upload CSV file for predictions
**Request**: Multipart form data with CSV file
**Response**: CSV file with predictions
**Required CSV Columns**:
- Pregnancies, Glucose, BloodPressure, SkinThickness
- Insulin, BMI, DiabetesPedigreeFunction, Age

### **MLflow Integration**

#### `GET /mlflow/experiments`
**Purpose**: List MLflow experiments and runs
**Response**:
```json
{
  "experiment": {
    "experiment_id": "0",
    "experiment_name": "diabetes_prediction",
    "lifecycle_stage": "active"
  },
  "recent_runs": [...]
}
```

#### `GET /mlflow/best-model`
**Purpose**: Get best model from MLflow registry
**Response**:
```json
{
  "run_id": "abc123",
  "accuracy": 0.8234,
  "f1_score": 0.7891,
  "start_time": "2025-12-06T20:30:15",
  "status": "FINISHED"
}
```

### **Model Management**

#### `POST /model/retrain`
**Purpose**: Trigger model retraining (background task)
**Response**:
```json
{
  "message": "Model retraining started",
  "status": "background_task_started"
}
```

## 🔧 **Input Validation**

### **DiabetesInput Schema**
```python
class DiabetesInput(BaseModel):
    pregnancies: int = Field(..., ge=0, le=20)
    glucose: float = Field(..., gt=0, le=300)
    blood_pressure: float = Field(..., gt=0, le=200)
    skin_thickness: float = Field(..., ge=0, le=100)
    insulin: float = Field(..., ge=0, le=1000)
    bmi: float = Field(..., gt=0, le=70)
    diabetes_pedigree_function: float = Field(..., gt=0, le=3)
    age: int = Field(..., gt=0, le=120)
```

### **Risk Level Calculation**
- **Low**: Probability < 0.3
- **Medium**: 0.3 ≤ Probability < 0.7
- **High**: Probability ≥ 0.7

## 🧪 **Testing the API**

### **1. Using cURL**
```bash
# Health check
curl -X GET "http://localhost:8000/health"

# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pregnancies": 2,
    "glucose": 120.0,
    "blood_pressure": 80.0,
    "skin_thickness": 25.0,
    "insulin": 100.0,
    "bmi": 25.5,
    "diabetes_pedigree_function": 0.5,
    "age": 35
  }'

# File upload
curl -X POST "http://localhost:8000/predict/file" \
  -F "file=@api_test_data.csv"
```

### **2. Using Python Client**
```bash
# Run comprehensive tests
python3 api_client.py --test

# Run downstream consumption demo
python3 api_client.py --demo

# Create sample test data
python3 api_client.py --create-data
```

### **3. Using API Documentation**
Visit http://localhost:8000/docs for interactive API testing

## 🏢 **Downstream Integration Examples**

### **Healthcare System Integration**
```python
import requests

# Initialize API client
api_url = "http://localhost:8000"

# Process patient queue
def process_patient_queue(patients):
    results = []
    for patient in patients:
        response = requests.post(
            f"{api_url}/predict",
            json=patient['data']
        )
        if response.status_code == 200:
            prediction = response.json()
            # Route based on risk level
            if prediction['risk_level'] == 'High':
                schedule_immediate_consultation(patient['id'])
            elif prediction['risk_level'] == 'Medium':
                schedule_followup(patient['id'])
            # Log results
            results.append({
                'patient_id': patient['id'],
                'prediction': prediction
            })
    return results
```

### **Batch Processing Service**
```python
# Process large datasets
def process_daily_screenings(csv_file_path):
    with open(csv_file_path, 'rb') as f:
        response = requests.post(
            f"{api_url}/predict/file",
            files={"file": f}
        )
    
    if response.status_code == 200:
        # Save results
        with open("screening_results.csv", "wb") as result_file:
            result_file.write(response.content)
        return "screening_results.csv"
```

### **Real-time Monitoring**
```python
# Monitor API health
def monitor_api_health():
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        health_data = response.json()
        
        if not health_data.get('model_loaded', False):
            send_alert("Model not loaded!")
        
        if health_data.get('status') != 'healthy':
            send_alert("API unhealthy!")
            
    except requests.RequestException:
        send_alert("API not responding!")
```

## 📊 **Performance & Scalability**

### **Current Capabilities**
- **Single Predictions**: ~100ms response time
- **Batch Processing**: Up to 1000 samples per request
- **File Processing**: Handles CSV files up to several MB
- **Concurrent Requests**: Async FastAPI handles multiple requests

### **Scaling Recommendations**
1. **Load Balancing**: Deploy multiple API instances behind a load balancer
2. **Caching**: Add Redis for model caching and response caching
3. **Database**: Move from file-based MLflow to database backend
4. **Container Orchestration**: Use Kubernetes for auto-scaling
5. **Monitoring**: Implement Prometheus/Grafana monitoring

## 🔒 **Security Considerations**

### **Current Security**
- Input validation with Pydantic
- File type validation for uploads
- Request size limits
- Error handling without data leakage

### **Production Security Enhancements**
```python
# Add authentication
from fastapi.security import HTTPBearer
security = HTTPBearer()

@app.post("/predict")
async def predict(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Validate token
    pass

# Add rate limiting
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.post("/predict")
@limiter.limit("100/minute")
async def predict():
    pass
```

## 🐛 **Error Handling**

### **HTTP Status Codes**
- **200**: Success
- **400**: Bad request (validation error)
- **503**: Service unavailable (model not loaded)
- **500**: Internal server error

### **Error Response Format**
```json
{
  "detail": "Validation error message",
  "message": "Additional context"
}
```

## 📋 **Deployment Checklist**

### **Pre-deployment**
- [ ] Model trained and saved
- [ ] All dependencies installed
- [ ] Configuration files present
- [ ] API tests passing

### **Production Deployment**
- [ ] Environment variables configured
- [ ] Load balancer configured
- [ ] SSL certificates installed
- [ ] Monitoring systems active
- [ ] Backup procedures in place

## 🛠 **Troubleshooting**

### **Common Issues**

1. **"Model not loaded" error**
   ```bash
   # Solution: Train a model first
   python3 train_simple.py
   ```

2. **"Port already in use" error**
   ```bash
   # Solution: Kill existing process or use different port
   lsof -ti:8000 | xargs kill -9
   ```

3. **"MLflow connection failed"**
   ```bash
   # Solution: Check MLflow configuration
   ls mlruns/
   ```

4. **File upload issues**
   - Ensure CSV has required columns
   - Check file size limits
   - Verify file encoding (UTF-8)

This FastAPI implementation provides a production-ready foundation for diabetes prediction services that can be easily consumed by downstream applications and scaled for enterprise use.