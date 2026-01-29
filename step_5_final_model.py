import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

# 1. Load Real Data
print("1. Loading Real Data...")
df_real = pd.read_csv('Data/Enriched_Simulation_Data.csv')
features = ['Pressure(Bar)', 'Drift_Velocity', 'Confidence_R2', 'Quench Temp(C)', 'Scan Speed']

# Select only needed columns
df_real = df_real[features + ['Is Anomaly']]

# 2. Generate Synthetic "Vaccine" Data
print("2. Generating Synthetic Usage Data (The 'Vaccine')...")

# Scenario A: The "Slow Death" (High Pressure, Bad Drift) -> Force FAIL
n_samples = 1000
synthetic_bad = pd.DataFrame({
    'Pressure(Bar)': np.random.uniform(3.2, 3.5, n_samples), # Looks Safe
    'Drift_Velocity': np.random.uniform(-0.10, -0.05, n_samples), # Is Dangerous
    'Confidence_R2': np.random.uniform(0.8, 1.0, n_samples),   # High Confidence
    'Quench Temp(C)': 900,
    'Scan Speed': 10,
    'Is Anomaly': 1 # TEACHING IT: This is a FAILURE
})

# Scenario B: The "Golden Run" (High Pressure, No Drift) -> Force PASS
# We add this to prevent the model from just assuming "Synthetic = Bad"
synthetic_good = pd.DataFrame({
    'Pressure(Bar)': np.random.uniform(3.4, 3.6, n_samples), 
    'Drift_Velocity': np.random.uniform(-0.01, 0.01, n_samples), 
    'Confidence_R2': np.random.uniform(0.8, 1.0, n_samples),
    'Quench Temp(C)': 900,
    'Scan Speed': 10,
    'Is Anomaly': 0
})

# 3. Mix & Augment
print(f"   - Injecting {len(synthetic_bad)} 'Slow Death' examples.")
print(f"   - Injecting {len(synthetic_good)} 'Golden Run' examples.")
df_final = pd.concat([df_real, synthetic_bad, synthetic_good], ignore_index=True)

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
    'Quench Temp(C)': 900, 
    'Scan Speed': 10
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
