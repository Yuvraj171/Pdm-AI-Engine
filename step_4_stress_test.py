import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

# 1. Load the Model and Data Schema
print("Loading the Machine Doctor...")
model = XGBClassifier()
model.load_model('final_machine_doctor.json')

# We need clear columns to match training
feature_names = ['Pressure(Bar)', 'Drift_Velocity', 'Confidence_R2', 'Quench Temp(C)', 'Scan Speed']

# 2. Define the 5 Test Cases (Synthetic Data)
# We create "Perfect" examples of each scenario to see what the AI thinks.

scenarios = [
    # TC-01: Golden Run (Everything perfect)
    {'ID': 'TC-01', 'Name': 'Golden Run',   'Pressure(Bar)': 3.5, 'Drift_Velocity': 0.00,  'Confidence_R2': 1.0, 'Quench Temp(C)': 900, 'Scan Speed': 10, 'Expected': 0},
    
    # TC-02: Hard Failure (Pressure dropped below 2.9)
    {'ID': 'TC-02', 'Name': 'Hard Failure', 'Pressure(Bar)': 2.0, 'Drift_Velocity': 0.00,  'Confidence_R2': 1.0, 'Quench Temp(C)': 900, 'Scan Speed': 10, 'Expected': 1},
    
    # TC-03: The Slow Death (Pressure OK, but Drifting Fast) -> THE TRAP
    {'ID': 'TC-03', 'Name': 'Slow Death',   'Pressure(Bar)': 3.2, 'Drift_Velocity': -0.06, 'Confidence_R2': 0.95, 'Quench Temp(C)': 900, 'Scan Speed': 10, 'Expected': 1},
    
    # TC-04: Noisy Sensor (Low R2)
    {'ID': 'TC-04', 'Name': 'Noisy Sensor', 'Pressure(Bar)': 3.0, 'Drift_Velocity': -0.06, 'Confidence_R2': 0.20, 'Quench Temp(C)': 900, 'Scan Speed': 10, 'Expected': 0},
    
    # TC-05: Context Fail (Low Temp + High Speed)
    {'ID': 'TC-05', 'Name': 'Context Fail', 'Pressure(Bar)': 3.0, 'Drift_Velocity': 0.00,  'Confidence_R2': 1.0, 'Quench Temp(C)': 700, 'Scan Speed': 12, 'Expected': 1},
]

# Convert to DataFrame
test_df = pd.DataFrame(scenarios)
X_test = test_df[feature_names]

# 3. Ask the Doctor (Predict)
print("\nRunning Stress Test...")
predictions = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1] # Probability of failure

# 4. Scorecard
print(f"{'ID':<6} | {'Scenario':<15} | {'Pressure':<8} | {'Drift':<8} | {'AI Prediction':<15} | {'Risk Score':<10} | {'Outcome'}")
print("-" * 90)

passed = 0
for i, row in test_df.iterrows():
    pred = predictions[i]
    prob = probs[i]
    expected = row['Expected']
    
    # Result Logic
    pred_str = "Failure (1)" if pred == 1 else "Healthy (0)"
    success = (pred == expected)
    outcome = "PASS" if success else "FAIL"
    if success: passed += 1
    
    print(f"{row['ID']:<6} | {row['Name']:<15} | {row['Pressure(Bar)']:<8} | {row['Drift_Velocity']:<8} | {pred_str:<15} | {prob:.1%}     | {outcome}")

print("-" * 90)
print(f"\nFinal Score: {passed}/5 Tests Passed.")

if predictions[2] == 0: # TC-03 check
    print("\nCRITICAL FINDING:")
    print("The model FAILED 'The Slow Death' test.")
    print("Reason: Pressure (3.2) is still in 'Safe Zone', so AI ignores the Drift (-0.06).")
    print("Correction Needed: We must RETRAIN with this scenario included.")
else:
    print("\nSUCCESS: Model correctly identified the Drift!")
