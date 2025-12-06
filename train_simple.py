"""
Simple training script for API testing (without MLflow)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

print("Starting simple model training...")

# Create model directory
os.makedirs("model", exist_ok=True)

# Load data
try:
    df = pd.read_csv("data/diabetes.csv")
    print(f"Loaded data with shape: {df.shape}")
except FileNotFoundError:
    print("Creating dummy data for testing...")
    import numpy as np
    np.random.seed(42)
    
    # Create synthetic diabetes data
    n_samples = 1000
    df = pd.DataFrame({
        'Pregnancies': np.random.randint(0, 10, n_samples),
        'Glucose': np.random.normal(120, 30, n_samples).clip(70, 200),
        'BloodPressure': np.random.normal(80, 15, n_samples).clip(50, 120),
        'SkinThickness': np.random.normal(25, 10, n_samples).clip(10, 50),
        'Insulin': np.random.normal(100, 50, n_samples).clip(0, 300),
        'BMI': np.random.normal(25, 8, n_samples).clip(15, 50),
        'DiabetesPedigreeFunction': np.random.uniform(0.1, 2.0, n_samples),
        'Age': np.random.randint(21, 80, n_samples),
        'Outcome': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    })
    print(f"Created synthetic data with shape: {df.shape}")

# Prepare features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model trained with accuracy: {accuracy:.4f}")

# Save model
model_path = "model/diabetes_model.pkl"
joblib.dump(model, model_path)
print(f"Model saved to: {model_path}")

print("Training completed successfully!")