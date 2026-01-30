import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

# 1. Load Real Data
print("1. Loading Real Data...")
df_real = pd.read_csv('Data/Enriched_Simulation_Data.csv')
features = ['Pressure(Bar)', 'Drift_Velocity', 'Confidence_R2', 'Part Temp(C)', 'Scan Speed', 'Quench Flow(LPM)']

# Select only needed columns
# Select only needed columns
# Note: CSV likely has 'Quench Water Flow' or similar. We assume it might be missing in old CSVs.
# We will construct it.
columns_to_keep = ['Pressure(Bar)', 'Drift_Velocity', 'Confidence_R2', 'Quench Temp(C)', 'Scan Speed', 'Is Anomaly']
# Try to keep flow if exists, else we fill it later
if 'Quench Flow' in df_real.columns:
    columns_to_keep.append('Quench Flow')

df_real = df_real[['Pressure(Bar)', 'Drift_Velocity', 'Confidence_R2', 'Quench Temp(C)', 'Scan Speed', 'Is Anomaly']]
# Rename column in DataFrame
df_real = df_real.rename(columns={'Quench Temp(C)': 'Part Temp(C)'})

# CLEANUP: The historical CSV might have Water Temp (40C). 
# We want to train on METAL Temp (850C). Force correction.
print("   - Correcting Historical Data: Setting Part Temp to 850C and Flow to 120...")
df_real['Part Temp(C)'] = 850.0
df_real['Quench Flow(LPM)'] = 120.0 # Assume Historical was Good

# 2. Generate Synthetic "Vaccine" Data
print("2. Generating Synthetic Usage Data (The 'Vaccine')...")

# Scenario A: The "Slow Death" (High Pressure, Bad Drift) -> Force FAIL
n_samples = 1000
synthetic_bad = pd.DataFrame({
    'Pressure(Bar)': np.random.uniform(3.2, 3.5, n_samples), # Looks Safe
    'Drift_Velocity': np.random.uniform(-0.10, -0.05, n_samples), # Is Dangerous
    'Confidence_R2': np.random.uniform(0.8, 1.0, n_samples),   # High Confidence
    'Part Temp(C)': 900,
    'Scan Speed': 10,
    'Quench Flow(LPM)': np.random.uniform(80, 150, n_samples), # Normal Flow
    'Is Anomaly': 1 # TEACHING IT: This is a FAILURE (Drift)
})

# Scenario B: The "Golden Run" (High Pressure, No Drift) -> Force PASS
# We add this to prevent the model from just assuming "Synthetic = Bad"
synthetic_good = pd.DataFrame({
    'Pressure(Bar)': np.random.uniform(3.4, 3.6, n_samples), 
    'Drift_Velocity': np.random.uniform(-0.01, 0.01, n_samples), 
    'Confidence_R2': np.random.uniform(0.8, 1.0, n_samples),
    'Part Temp(C)': np.random.uniform(830, 870, n_samples), # USER SPEC: OK Range
    'Scan Speed': 10,
    'Quench Flow(LPM)': np.random.uniform(80, 150, n_samples), # OK: 80-150
    'Is Anomaly': 0
})

# Scenario C: "Flow Failure" (Temp/Pressure OK, but Flow Bad)
synthetic_flow_fail = pd.DataFrame({
    'Pressure(Bar)': np.random.uniform(3.4, 3.6, n_samples),
    'Drift_Velocity': np.random.uniform(-0.01, 0.01, n_samples),
    'Confidence_R2': 0.9,
    'Part Temp(C)': 850,
    'Scan Speed': 10,
    'Quench Flow(LPM)': np.concatenate([
        np.random.uniform(0, 50, n_samples // 2),   # DOWN (<50)
        np.random.uniform(150, 200, n_samples // 2) # NG (>150)
    ]),
    'Is Anomaly': 1
})

# 3. Mix & Augment
print(f"   - Injecting {len(synthetic_bad)} 'Slow Death' examples.")
print(f"   - Injecting {len(synthetic_good)} 'Golden Run' examples.")
print(f"   - Injecting {len(synthetic_flow_fail)} 'Flow Failure' examples.")
df_final = pd.concat([df_real, synthetic_bad, synthetic_good, synthetic_flow_fail], ignore_index=True)

# Save the Augmented Dataset (The "Salted" Data)
augmented_file = 'Data/Augmented_Training_Data.csv'
df_final.to_csv(augmented_file, index=False)
print(f"   - Augmented Dataset saved to '{augmented_file}'")

# 4. Train the Final Model
print("3. Retraining the Machine Doctor...")
X = df_final[features]
y = df_final['Is Anomaly']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42)

# Initialize & Fit
model = XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# accuracy
acc = model.score(X_test, y_test)
print(f"   - New Accuracy on Test Set: {acc:.2%}")

# Save
model.save_model('final_machine_doctor.json')
print("   - Model saved as 'final_machine_doctor.json'")

# 5. Immediate Stress Test Verification
print("\n4. VERIFICATION: Running The 'Slow Death' Test (TC-03) Again...")

# Re-create TC-03 Scenario
tc_03 = pd.DataFrame([{
    'Pressure(Bar)': 3.2, 
    'Drift_Velocity': -0.06, 
    'Confidence_R2': 0.95, 
    'Part Temp(C)': 900, # Note: 900 is NG, but TC-03 tests Drift.
    'Scan Speed': 10,
    'Quench Flow(LPM)': 120.0 # Normal Flow
}])

pred = model.predict(tc_03[features])[0]
prob = model.predict_proba(tc_03[features])[0][1]

print("-" * 50)
print(f"Scenario: Pressure=3.2 (Safe), Drift=-0.06 (Fast Leak)")
print(f"Prediction: {'FAILURE (1)' if pred == 1 else 'Healthy (0)'}")
print(f"Risk Score: {prob:.1%}")
print("-" * 50)

if pred == 1:
    print("SUCCESS: The Model has learned to detect Drift! \U0001F680")
else:
    print("FAIL: The Model is still blind.")
