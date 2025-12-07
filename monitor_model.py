import pandas as pd
import os
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

os.makedirs("reports", exist_ok=True)

print("Loading reference data...")
reference_data = pd.read_csv("data/diabetes.csv")

current_data_path = "data/new_data.csv"
if os.path.exists(current_data_path):
    print(f"Loading current data from {current_data_path}...")
    current_data = pd.read_csv(current_data_path)
else:
    print("new_data.csv not found. Using a sample from reference data for demo...")
    current_data = reference_data.sample(frac=0.3, random_state=42).copy()

print(f"Reference data shape: {reference_data.shape}")
print(f"Current data shape: {current_data.shape}")

numerical_features = [
    "Pregnancies", "Glucose", "BloodPressure", 
    "SkinThickness", "Insulin", "BMI", 
    "DiabetesPedigreeFunction", "Age"
]

common_columns = list(set(reference_data.columns) & set(current_data.columns))
print(f"Common columns: {common_columns}")

reference_for_drift = reference_data[common_columns].copy()
current_for_drift = current_data[[col for col in common_columns if col in current_data.columns]].copy()

available_features = [f for f in numerical_features if f in common_columns]

column_mapping = ColumnMapping(
    numerical_features=available_features
)

print("\nGenerating Data Drift Report...")
data_drift_report = Report(metrics=[
    DataDriftPreset(),
])

data_drift_report.run(
    reference_data=reference_for_drift,
    current_data=current_for_drift,
    column_mapping=column_mapping
)

drift_report_path = "reports/data_drift_report.html"
data_drift_report.save_html(drift_report_path)
print(f"✅ Data drift report saved to {drift_report_path}")

print("\nGenerating Data Quality Report...")
quality_report = Report(metrics=[
    DataQualityPreset(),
])

quality_report.run(
    reference_data=reference_for_drift,
    current_data=current_for_drift,
    column_mapping=column_mapping
)

quality_report_path = "reports/data_quality_report.html"
quality_report.save_html(quality_report_path)
print(f"✅ Data quality report saved to {quality_report_path}")

print("\n" + "="*50)
print("Monitoring reports generated successfully!")
print("="*50)
print(f"\nOpen the reports in your browser:")
print(f"  - Data Drift: {os.path.abspath(drift_report_path)}")
print(f"  - Data Quality: {os.path.abspath(quality_report_path)}")
print("\nOr run: Start-Process reports/data_drift_report.html")
