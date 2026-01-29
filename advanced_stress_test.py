import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

print("--- MACHINE DOCTOR: ADVANCED STRESS TEST (Hypothesis Verification) ---")

# 1. Load Models to replicate Deployment
print("\n1. Loading Models...")
try:
    xgb_model = XGBClassifier()
    xgb_model.load_model('final_machine_doctor.json')
    
    rf_model = joblib.load('final_random_forest.joblib')
    print("   CHECK: Models loaded.")
except Exception as e:
    print(f"   ERROR: {e}")
    exit()

# 2. Define the "Double Check" Logic
def evaluate(scenario_name, data):
    with open("advanced_test_results.txt", "a") as f:
        f.write(f"\nRunning: {scenario_name}\n")
        f.write(f"   Input: Pressure={data['Pressure(Bar)']}, Drift={data['Drift_Velocity']}\n")
        
        input_df = pd.DataFrame([data])[['Pressure(Bar)', 'Drift_Velocity', 'Confidence_R2', 'Quench Temp(C)', 'Scan Speed']]
        
        # XGBoost
        xgb_pred = xgb_model.predict(input_df)[0]
        xgb_prob = xgb_model.predict_proba(input_df)[0][1]
        
        # Random Forest
        rf_pred = rf_model.predict(input_df)[0]
        rf_prob = rf_model.predict_proba(input_df)[0][1]
        
        f.write(f"   XGBoost Decision: {'FAIL (1)' if xgb_pred==1 else 'Safe (0)'} (Risk: {xgb_prob:.1%})\n")
        f.write(f"   RandFor Decision: {'FAIL (1)' if rf_pred==1 else 'Safe (0)'} (Risk: {rf_prob:.1%})\n")
        
        # Interpretation
        if xgb_pred == 1 and rf_pred == 1:
            f.write("   -> FINAL ACTION: EMERGENCY STOP (Critical Failure)\n")
        elif xgb_pred == 1 and rf_pred == 0:
            f.write("   -> FINAL ACTION: WARNING (Possible Early Drift)\n")
        else:
            f.write("   -> FINAL ACTION: NONE (System Nominal)\n")

# Clear file
open("advanced_test_results.txt", "w").close()

# 3. Create Scenarios

# TC-06: Recovery (Hysteresis)
# "Pressure dropped to 3.2 (so it WAS failing), but Drift is now 0.0 (it stopped leaking)."
# EXPECTATION: Safe (or Warning). Should NOT be Critical.
tc_06 = {
    'Pressure(Bar)': 3.2, 
    'Drift_Velocity': 0.00,  # STABLE
    'Confidence_R2': 1.0, 
    'Quench Temp(C)': 900, 
    'Scan Speed': 10
}

# TC-07: Sensor Freeze (Flatline)
# "Pressure is 3.5 (Normal), Drift is 0.0."
# EXPECTATION: Safe. (Unless we trained it to detect 'Frozen Sensor', it interprets this as 'Perfect operation').
tc_07 = {
    'Pressure(Bar)': 3.5, 
    'Drift_Velocity': 0.00, 
    'Confidence_R2': 1.0, 
    'Quench Temp(C)': 900, 
    'Scan Speed': 10
}

# TC-08: Start-up Transient (Positive Drift)
# "Pressure is Low (1.5), but Drift is POSITIVE (+0.2) (Turning On)."
# EXPECTATION: Safe. (Drift is wrong direction for failure).
tc_08 = {
    'Pressure(Bar)': 1.5, 
    'Drift_Velocity': 0.20,  # RISING FAST
    'Confidence_R2': 0.95, 
    'Quench Temp(C)': 900, 
    'Scan Speed': 10
}

# 4. Execute
evaluate("TC-06: The Recovery (Low Pressure, Stable Drift)", tc_06)
evaluate("TC-07: Sensor Freeze (Normal Pressure, Zero Drift)", tc_07)
evaluate("TC-08: Start-up Transient (Low Pressure, Positive Drift)", tc_08)
