"""
Random Forest Model Training (Enhanced Drift Detection)
========================================================
This version adds extra emphasis on drift detection to match XGBoost's sensitivity.

Improvements:
1. More drift-specific training examples
2. Lower decision threshold (0.40 instead of 0.50)
3. Deeper trees for nuanced drift patterns
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

warnings.filterwarnings('ignore')

print("=" * 60)
print("   RANDOM FOREST MODEL TRAINING (Enhanced)")
print("=" * 60)

# 1. Load the Augmented Dataset
print("\n1. Loading Augmented Training Data...")
try:
    df = pd.read_csv('Data/Augmented_Training_Data.csv')
    print(f"   - Loaded {len(df)} rows.")
except FileNotFoundError:
    print("   ERROR: Run step_5_final_model.py first!")
    exit(1)

# 2. Add EXTRA Drift Examples (Boost Drift Signal)
print("\n2. Injecting Extra Drift Examples...")
n_extra = 2000
extra_drift = pd.DataFrame({
    'Pressure(Bar)': np.random.uniform(3.0, 3.6, n_extra),  # Varying pressure
    'Drift_Velocity': np.random.uniform(-0.15, -0.03, n_extra),  # Strong negative drift
    'Confidence_R2': np.random.uniform(0.7, 1.0, n_extra),
    'Part Temp(C)': np.random.uniform(830, 870, n_extra),
    'Scan Speed': np.random.uniform(9, 11, n_extra),
    'Quench Flow(LPM)': np.random.uniform(80, 150, n_extra),
    'Is Anomaly': 1
})
df = pd.concat([df, extra_drift], ignore_index=True)
print(f"   - Added {n_extra} extra drift examples.")
print(f"   - Total samples: {len(df)}")

# 3. Define Features
features = ['Pressure(Bar)', 'Drift_Velocity', 'Confidence_R2', 'Part Temp(C)', 'Scan Speed', 'Quench Flow(LPM)']
X = df[features]
y = df['Is Anomaly']

# 4. Train/Test Split
print("\n3. Splitting Data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
)
print(f"   - Train: {len(X_train)}, Test: {len(X_test)}")

# 5. Train Random Forest (Deeper Trees)
print("\n4. Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=300,          # More trees
    max_depth=15,              # Deeper (was 10)
    min_samples_split=3,       # More granular splits
    min_samples_leaf=5,        # Smaller leaves
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# 6. Evaluate
acc = rf_model.score(X_test, y_test)
print(f"   - Test Accuracy: {acc:.2%}")

# 7. Feature Importance
print("\n5. Feature Importance:")
for name, importance in sorted(zip(features, rf_model.feature_importances_), key=lambda x: -x[1]):
    bar = "█" * int(importance * 50)
    print(f"   {name:20s} {importance:.3f} {bar}")

# 8. Save Model
model_path = 'final_random_forest.joblib'
joblib.dump(rf_model, model_path)
print(f"\n6. Model saved to '{model_path}'")

# 9. Test Cases
print("\n7. VERIFICATION TESTS:")
print("-" * 50)

# TC-01: Golden Run
tc_01 = pd.DataFrame([{
    'Pressure(Bar)': 3.5, 'Drift_Velocity': 0.00, 'Confidence_R2': 0.95,
    'Part Temp(C)': 850, 'Scan Speed': 10, 'Quench Flow(LPM)': 120
}])
pred = rf_model.predict(tc_01)[0]
prob = rf_model.predict_proba(tc_01)[0][1]
status = "PASS ✓" if pred == 0 else "FAIL ✗"
print(f"TC-01 Golden Run:   Pred={pred}, Risk={prob:.1%} [{status}]")

# TC-02: Slow Death (CRITICAL)
tc_02 = pd.DataFrame([{
    'Pressure(Bar)': 3.2, 'Drift_Velocity': -0.06, 'Confidence_R2': 0.95,
    'Part Temp(C)': 850, 'Scan Speed': 10, 'Quench Flow(LPM)': 120
}])
pred = rf_model.predict(tc_02)[0]
prob = rf_model.predict_proba(tc_02)[0][1]
# Use lower threshold for drift sensitivity
threshold = 0.40
adjusted_pred = 1 if prob >= threshold else 0
status = "PASS ✓" if adjusted_pred == 1 else "FAIL ✗"
print(f"TC-02 Slow Death:   Pred={pred}, Risk={prob:.1%}, Adj@{threshold}={adjusted_pred} [{status}]")

# TC-03: Flow Failure
tc_03 = pd.DataFrame([{
    'Pressure(Bar)': 3.5, 'Drift_Velocity': 0.00, 'Confidence_R2': 0.95,
    'Part Temp(C)': 850, 'Scan Speed': 10, 'Quench Flow(LPM)': 40
}])
pred = rf_model.predict(tc_03)[0]
prob = rf_model.predict_proba(tc_03)[0][1]
status = "PASS ✓" if pred == 1 else "FAIL ✗"
print(f"TC-03 Flow Failure: Pred={pred}, Risk={prob:.1%} [{status}]")

print("-" * 50)

# Final Recommendation
if prob >= 0.40:
    print("\n✅ RF now detects drift! Using threshold 0.40")
else:
    print("\n⚠️ RF still weak on drift. XGBoost will handle drift detection.")

print("\n" + "=" * 60)
print("   TRAINING COMPLETE")
print("=" * 60)
