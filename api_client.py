"""
Test client for Diabetes Prediction API

This script demonstrates how to consume the FastAPI endpoints
"""

import requests
import json
import pandas as pd
import time
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"

class DiabetesAPIClient:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        
    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        try:
            response = requests.get(f"{self.base_url}/model/info")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def predict_single(self, patient_data: Dict[str, Any], log_to_mlflow: bool = True) -> Dict[str, Any]:
        """Make a single prediction"""
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json=patient_data,
                params={"log_to_mlflow": log_to_mlflow}
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def predict_batch(self, patients_data: list) -> Dict[str, Any]:
        """Make batch predictions"""
        try:
            payload = {"samples": patients_data}
            response = requests.post(
                f"{self.base_url}/predict/batch",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def predict_from_file(self, file_path: str) -> str:
        """Upload file for predictions"""
        try:
            with open(file_path, 'rb') as f:
                files = {"file": (file_path, f, "text/csv")}
                response = requests.post(
                    f"{self.base_url}/predict/file",
                    files=files
                )
                response.raise_for_status()
                
                # Save response content as file
                result_filename = f"api_predictions_{int(time.time())}.csv"
                with open(result_filename, 'wb') as result_file:
                    result_file.write(response.content)
                
                return result_filename
        except requests.RequestException as e:
            return f"Error: {str(e)}"
        except FileNotFoundError:
            return f"Error: File {file_path} not found"
    
    def get_mlflow_experiments(self) -> Dict[str, Any]:
        """Get MLflow experiment information"""
        try:
            response = requests.get(f"{self.base_url}/mlflow/experiments")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def get_best_model(self) -> Dict[str, Any]:
        """Get best model information from MLflow"""
        try:
            response = requests.get(f"{self.base_url}/mlflow/best-model")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def trigger_retraining(self) -> Dict[str, Any]:
        """Trigger model retraining"""
        try:
            response = requests.post(f"{self.base_url}/model/retrain")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

def test_api_endpoints():
    """Test all API endpoints"""
    client = DiabetesAPIClient()
    
    print("=== Diabetes Prediction API Test ===\n")
    
    # Test health check
    print("1. Testing health check...")
    health = client.health_check()
    print(f"Health: {json.dumps(health, indent=2)}\n")
    
    if health.get("error"):
        print("API is not running. Start the API with: uvicorn api:app --reload")
        return
    
    # Test model info
    print("2. Testing model info...")
    model_info = client.get_model_info()
    print(f"Model Info: {json.dumps(model_info, indent=2)}\n")
    
    # Test single prediction
    print("3. Testing single prediction...")
    sample_patient = {
        "pregnancies": 2,
        "glucose": 120.0,
        "blood_pressure": 80.0,
        "skin_thickness": 25.0,
        "insulin": 100.0,
        "bmi": 25.5,
        "diabetes_pedigree_function": 0.5,
        "age": 35
    }
    
    prediction = client.predict_single(sample_patient)
    print(f"Single Prediction: {json.dumps(prediction, indent=2)}\n")
    
    # Test batch prediction
    print("4. Testing batch prediction...")
    batch_patients = [
        sample_patient,
        {
            "pregnancies": 0,
            "glucose": 95.0,
            "blood_pressure": 70.0,
            "skin_thickness": 20.0,
            "insulin": 80.0,
            "bmi": 22.0,
            "diabetes_pedigree_function": 0.3,
            "age": 25
        },
        {
            "pregnancies": 5,
            "glucose": 180.0,
            "blood_pressure": 90.0,
            "skin_thickness": 35.0,
            "insulin": 200.0,
            "bmi": 35.0,
            "diabetes_pedigree_function": 1.2,
            "age": 45
        }
    ]
    
    batch_prediction = client.predict_batch(batch_patients)
    print(f"Batch Prediction: {json.dumps(batch_prediction, indent=2)}\n")
    
    # Test MLflow endpoints
    print("5. Testing MLflow integration...")
    experiments = client.get_mlflow_experiments()
    print(f"MLflow Experiments: {json.dumps(experiments, indent=2)}\n")
    
    best_model = client.get_best_model()
    print(f"Best Model: {json.dumps(best_model, indent=2)}\n")
    
    # Test file prediction (if test file exists)
    test_file = "data/test.csv"
    if pd.read_csv(test_file).shape[0] > 0:
        print("6. Testing file prediction...")
        result_file = client.predict_from_file(test_file)
        print(f"File prediction result saved to: {result_file}\n")
    
    print("=== API Test Complete ===")

def create_sample_test_data():
    """Create sample test data for file upload"""
    sample_data = pd.DataFrame({
        'Pregnancies': [1, 0, 3, 2],
        'Glucose': [110, 95, 150, 120],
        'BloodPressure': [75, 70, 85, 80],
        'SkinThickness': [22, 18, 30, 25],
        'Insulin': [90, 70, 150, 100],
        'BMI': [23.5, 21.0, 32.0, 25.5],
        'DiabetesPedigreeFunction': [0.4, 0.2, 0.8, 0.5],
        'Age': [28, 22, 40, 35]
    })
    
    sample_data.to_csv("sample_test_data.csv", index=False)
    print("Sample test data created: sample_test_data.csv")
    return "sample_test_data.csv"

def demo_downstream_consumption():
    """Demonstrate how downstream services can consume the API"""
    client = DiabetesAPIClient()
    
    print("=== Downstream Service Demo ===\n")
    
    # Simulate a healthcare system using the API
    patients_queue = [
        {"id": "P001", "data": {"pregnancies": 2, "glucose": 120, "blood_pressure": 80, "skin_thickness": 25, "insulin": 100, "bmi": 25.5, "diabetes_pedigree_function": 0.5, "age": 35}},
        {"id": "P002", "data": {"pregnancies": 0, "glucose": 95, "blood_pressure": 70, "skin_thickness": 20, "insulin": 80, "bmi": 22.0, "diabetes_pedigree_function": 0.3, "age": 25}},
        {"id": "P003", "data": {"pregnancies": 5, "glucose": 180, "blood_pressure": 90, "skin_thickness": 35, "insulin": 200, "bmi": 35.0, "diabetes_pedigree_function": 1.2, "age": 45}}
    ]
    
    # Process each patient
    results = []
    for patient in patients_queue:
        print(f"Processing patient {patient['id']}...")
        
        prediction = client.predict_single(patient['data'], log_to_mlflow=True)
        
        if "error" not in prediction:
            result = {
                "patient_id": patient['id'],
                "prediction": prediction['prediction'],
                "probability": prediction['probability'],
                "risk_level": prediction['risk_level'],
                "recommended_action": get_recommendation(prediction)
            }
            results.append(result)
            print(f"  Result: {result['recommended_action']}")
        else:
            print(f"  Error: {prediction['error']}")
    
    print(f"\nProcessed {len(results)} patients successfully")
    
    # Generate summary report
    high_risk_count = sum(1 for r in results if r['risk_level'] == 'High')
    print(f"High-risk patients identified: {high_risk_count}")
    
    return results

def get_recommendation(prediction):
    """Get recommendation based on prediction"""
    if prediction['prediction'] == 1:
        if prediction['risk_level'] == 'High':
            return "Immediate consultation recommended"
        else:
            return "Schedule follow-up appointment"
    else:
        return "Continue regular health monitoring"

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diabetes API Test Client")
    parser.add_argument("--test", action="store_true", help="Run API endpoint tests")
    parser.add_argument("--demo", action="store_true", help="Run downstream consumption demo")
    parser.add_argument("--create-data", action="store_true", help="Create sample test data")
    
    args = parser.parse_args()
    
    if args.create_data:
        create_sample_test_data()
    elif args.test:
        test_api_endpoints()
    elif args.demo:
        demo_downstream_consumption()
    else:
        print("Use --test to run API tests, --demo for downstream demo, or --create-data for sample data")
        print("Example: python api_client.py --test")