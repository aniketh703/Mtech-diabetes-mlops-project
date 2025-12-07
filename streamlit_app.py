import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import json

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Diabetes Prediction Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .prediction-positive {
        background-color: #ffcccc;
        border-left: 5px solid #ff0000;
        padding: 1rem;
        border-radius: 5px;
    }
    .prediction-negative {
        background-color: #ccffcc;
        border-left: 5px solid #00ff00;
        padding: 1rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def make_prediction(data: dict):
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=data,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API returned status {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def main():
    st.markdown('<h1 class="main-header">🏥 Diabetes Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["🔮 Make Prediction", "📊 Batch Prediction", "📈 Model Info", "ℹ️ About"]
    )
    
    api_status = check_api_health()
    if api_status:
        st.sidebar.success("✅ API Connected")
    else:
        st.sidebar.error("❌ API Offline")
        st.warning("⚠️ The prediction API is not available. Please ensure the API server is running.")
    
    if page == "🔮 Make Prediction":
        show_prediction_page()
    elif page == "📊 Batch Prediction":
        show_batch_prediction_page()
    elif page == "📈 Model Info":
        show_model_info_page()
    else:
        show_about_page()


def show_prediction_page():
    st.header("🔮 Single Patient Prediction")
    
    st.markdown("""
    Enter the patient's health metrics below to predict diabetes risk.
    All values should be within normal clinical ranges.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0, max_value=20, value=1,
            help="Number of times pregnant"
        )
        glucose = st.number_input(
            "Glucose (mg/dL)",
            min_value=0.0, max_value=300.0, value=120.0,
            help="Plasma glucose concentration"
        )
        blood_pressure = st.number_input(
            "Blood Pressure (mmHg)",
            min_value=0.0, max_value=200.0, value=70.0,
            help="Diastolic blood pressure"
        )
        skin_thickness = st.number_input(
            "Skin Thickness (mm)",
            min_value=0.0, max_value=100.0, value=20.0,
            help="Triceps skin fold thickness"
        )
    
    with col2:
        insulin = st.number_input(
            "Insulin (μU/mL)",
            min_value=0.0, max_value=1000.0, value=80.0,
            help="2-Hour serum insulin"
        )
        bmi = st.number_input(
            "BMI",
            min_value=0.0, max_value=70.0, value=25.0,
            help="Body mass index"
        )
        dpf = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0, max_value=3.0, value=0.5,
            help="Diabetes pedigree function score"
        )
        age = st.number_input(
            "Age (years)",
            min_value=1, max_value=120, value=30,
            help="Age in years"
        )
    
    if st.button("🔍 Predict", type="primary", use_container_width=True):
        data = {
            "pregnancies": pregnancies,
            "glucose": glucose,
            "blood_pressure": blood_pressure,
            "skin_thickness": skin_thickness,
            "insulin": insulin,
            "bmi": bmi,
            "diabetes_pedigree_function": dpf,
            "age": age
        }
        
        with st.spinner("Making prediction..."):
            result = make_prediction(data)
        
        if "error" in result:
            st.error(f"❌ Error: {result['error']}")
        else:
            st.success("✅ Prediction Complete!")
            
            col1, col2, col3 = st.columns(3)
            
            prediction = result.get("prediction", 0)
            probability = result.get("probability", 0.5)
            risk_level = result.get("risk_level", "Unknown")
            
            with col1:
                st.metric("Prediction", "Diabetic" if prediction == 1 else "Non-Diabetic")
            with col2:
                st.metric("Probability", f"{probability:.1%}")
            with col3:
                st.metric("Risk Level", risk_level)
            
            if prediction == 1:
                st.markdown("""
                <div class="prediction-positive">
                    <h3>⚠️ High Risk Detected</h3>
                    <p>Based on the provided metrics, there is an elevated risk of diabetes. 
                    Please consult with a healthcare professional for proper evaluation.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="prediction-negative">
                    <h3>✅ Low Risk</h3>
                    <p>Based on the provided metrics, the risk of diabetes appears low. 
                    Continue maintaining a healthy lifestyle.</p>
                </div>
                """, unsafe_allow_html=True)


def show_batch_prediction_page():
    st.header("📊 Batch Prediction")
    
    st.markdown("""
    Upload a CSV file with patient data for batch predictions.
    The file should contain the following columns:
    - `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`
    - `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of uploaded data:")
        st.dataframe(df.head(10))
        
        if st.button("🔍 Run Batch Prediction", type="primary"):
            st.info("Batch prediction feature - connect to API endpoint for full functionality")


def show_model_info_page():
    st.header("📈 Model Information")
    
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=5)
        if response.status_code == 200:
            info = response.json()
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Model Details")
                st.json(info)
            with col2:
                st.subheader("Performance Metrics")
                if "accuracy" in info:
                    st.metric("Accuracy", f"{info['accuracy']:.2%}")
    except:
        st.info("Model information unavailable. Please ensure the API is running.")
        
        st.subheader("Expected Model Features")
        features = [
            "Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness",
            "Insulin", "BMI", "Diabetes Pedigree Function", "Age"
        ]
        for i, feat in enumerate(features, 1):
            st.write(f"{i}. {feat}")


def show_about_page():
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ## 🏥 MLOps Diabetes Prediction Project
    
    This is a comprehensive MLOps pipeline for diabetes prediction using machine learning.
    
    ### 🔧 Technology Stack
    - **Machine Learning**: scikit-learn, pandas, numpy
    - **MLOps**: MLflow for experiment tracking
    - **Data Versioning**: DVC (Data Version Control)
    - **API Framework**: FastAPI
    - **Dashboard**: Streamlit
    - **Containerization**: Docker
    - **CI/CD**: GitHub Actions
    
    ### 📊 Dataset
    The model is trained on the Pima Indians Diabetes Database, which contains various 
    health metrics for female patients of Pima Indian heritage.
    
    ### ⚠️ Disclaimer
    This tool is for educational purposes only and should not be used as a substitute 
    for professional medical advice, diagnosis, or treatment.
    """)


if __name__ == "__main__":
    main()
