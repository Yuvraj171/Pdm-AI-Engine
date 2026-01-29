import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

print("--- MACHINE DOCTOR DEPLOYMENT TEST ---")

# 1. Load Models
print("\n1. Loading Models...")
try:
    # Load XGBoost (The Paramedic)
    xgb_model = XGBClassifier()
    xgb_model.load_model('final_machine_doctor.json')
    print("   CHECK: XGBoost loaded successfully.")

    # Load Random Forest (The Surgeon)
    rf_model = joblib.load('final_random_forest.joblib')
    print("   CHECK: Random Forest loaded successfully.")
except Exception as e:
    print(f"   ERROR: Could not load models. Did you run the training notebooks? {e}")
    exit()

# 2. Define The Logic
def machine_doctor_check(sensor_input):
    """
    The 'Two Doctor' Protocol:
    1. Ask XGBoost (Fast, Sensitive).
    2. If XGBoost alarms, ask Random Forest (Precise, Robust).
    """
    
    # Preprocessing (column names must match training)
    features = ['Pressure(Bar)', 'Drift_Velocity', 'Confidence_R2', 'Quench Temp(C)', 'Scan Speed']
    input_df = pd.DataFrame([sensor_input])[features]

    # --- STEP 1: XGBoost Scan ---
    xgb_pred = xgb_model.predict(input_df)[0]
    xgb_prob = xgb_model.predict_proba(input_df)[0][1]

    if xgb_pred == 0:
        return "SAFE", f"XGBoost says OK ({xgb_prob:.1%} risk)"

    # --- STEP 2: Random Forest Verify (Only if XGBoost alarms) ---
    print(f"   [!] XGBoost detected anomaly (Risk: {xgb_prob:.1%}). Verifying with Random Forest...")
    
    rf_pred = rf_model.predict(input_df)[0]
    rf_prob = rf_model.predict_proba(input_df)[0][1]

    if rf_pred == 1:
        return "CRITICAL_FAILURE", f"CONFIRMED. Both models agree. (RF Risk: {rf_prob:.1%})"
    else:
        return "WARNING_DRIFT", f"DISAGREEMENT. XGBoost=Fail, RF=Safe. Likely Early Drift."

# 3. Validation Scenarios
print("\n2. Running Verification Scenarios...")

# Scenario A: Golden Run (Should be SAFE)
test_a = {
    'Pressure(Bar)': 3.5, 
    'Drift_Velocity': 0.00, 
    'Confidence_R2': 1.0, 
    'Quench Temp(C)': 900, 
    'Scan Speed': 10
}

# Scenario B: Slow Death (TC-03) (Should be CRITICAL)
test_b = {
    'Pressure(Bar)': 3.2, 
    'Drift_Velocity': -0.06, 
    'Confidence_R2': 0.95, 
    'Quench Temp(C)': 900, 
    'Scan Speed': 10
}

# Scenario C: Noise (Should be WARNING/SAFE depending on threshold)
test_c = {
    'Pressure(Bar)': 3.4, 
    'Drift_Velocity': -0.02, # Minor drift
    'Confidence_R2': 0.60, # Low confidence
    'Quench Temp(C)': 900, 
    'Scan Speed': 10
}

# Scenario D: Dashboard Zero State (The "Empty Buffer" Panic)
test_d = {
    'Pressure(Bar)': 0.0,   # SENSOR OFF / NOT CONNECTED
    'Drift_Velocity': 0.0,  # NO HISTORY
    'Confidence_R2': 0.0,   # NO CONFIDENCE
    'Quench Temp(C)': 0,    # OFF
    'Scan Speed': 0         # OFF
}

scenarios = [("Golden Run", test_a), ("Slow Death", test_b), ("Noisy Data", test_c), ("Dashboard Zero State", test_d)]

for name, data in scenarios:
    print(f"\nTesting: {name}")
    status, message = machine_doctor_check(data)
    print(f"RESULT: {status}")
    print(f"REASON: {message}")
