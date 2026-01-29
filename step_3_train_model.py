import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np

# 1. Load the "Enriched" data (from Step 2)
# Using forward slash works on Windows and avoids escape char errors
file_path = 'Data/Enriched_Simulation_Data.csv'
print(f"Loading data from {file_path}...")
df = pd.read_csv(file_path)

# 2. Define our "Features" (The inputs) and "Target" (The answer)
# X = The Clues (Pressure, Velocity, Temp)
# y = The Result (Is it an Anomaly? 0 or 1)
features = ['Pressure(Bar)', 'Drift_Velocity', 'Confidence_R2', 'Quench Temp(C)', 'Scan Speed']

X = df[features]
y = df['Is Anomaly']

print("Features selected:", features)

# 3. Split the data (The Correction)
# Problem: Old method gave 32% failure in Train but only 3% in Test (Unfair exam).
# Fix: 'stratify=y' ensures both sets have exactly the same % of failures.
print("Splitting data (Stratified)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42)

print(f"Training on {len(X_train)} rows.")
print(f"Testing on {len(X_test)} rows.")

# --- BLOCK 2: Initialize the Brain ---
print("Initializing XGBoost Model...")
# Removed 'scale_pos_weight=10' because Stratified Split gives us balanced exposure
model = XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')


# --- BLOCK 3: The Training (Study) and The Exam ---
print("Training the model...")
model.fit(X_train, y_train)

print("Evaluating performance...")
# This gives us the "Grade" on the exam
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy on Test Data: {accuracy:.2%}")

# Optional: Show what the model thinks is important
print("\nFeature Importance:")
for name, importance in zip(features, model.feature_importances_):
    print(f"{name}: {importance:.4f}")

# 5. Save the Model (For Step 4)
model.save_model('machine_doctor.json')
print("\nModel saved as 'machine_doctor.json'. Ready for Stress Testing.")
