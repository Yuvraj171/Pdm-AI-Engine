import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

print("--- MODEL HEALTH CHECK ---")

# 1. Load the 'Salted' Data (Best representation of reality + edge cases)
df = pd.read_csv('Data/Augmented_Training_Data.csv')
features = ['Pressure(Bar)', 'Drift_Velocity', 'Confidence_R2', 'Quench Temp(C)', 'Scan Speed']
X = df[features]
y = df['Is Anomaly']

# 2. Split (Stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42)

# 3. Load Model
model = XGBClassifier()
model.load_model('final_machine_doctor.json')

# CHECK 1: The "Gap Test" (Overfitting)
# Logic: If Train Score >> Test Score, you are overfitting (memorizing).
print("\n[Check 1] Overfitting Analysis:")
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
gap = train_score - test_score

print(f"   Train Accuracy: {train_score:.2%}")
print(f"   Test Accuracy:  {test_score:.2%} (The Real World Score)")
print(f"   Gap:            {gap:.2%}")

if gap > 0.05: # >5% Gap
    print("   RESULT: WARNING. Possible Overfitting. Model is memorizing the textbook.")
elif gap < -0.02:
    print("   RESULT: STRANGE. Underfitting or Data Leakage.")
else:
    print("   RESULT: PASS. Model generalizes well.")

# CHECK 2: Confusion Matrix (The "Bias" Test)
# Logic: Are we catching all failures? Or missing them?
print("\n[Check 2] Confusion Matrix:")
preds = model.predict(X_test)
cm = confusion_matrix(y_test, preds)
tn, fp, fn, tp = cm.ravel()

print(f"   True Healthy  (TN): {tn} (Predicted Safe, Was Safe)")
print(f"   False Alarm   (FP): {fp} (Predicted Fail, Was Safe) -> Annoying but Safe")
print(f"   Missed Fail   (FN): {fn} (Predicted Safe, Was Fail) -> DANGEROUS!")
print(f"   True Fail     (TP): {tp} (Predicted Fail, Was Fail)")

if fn > (tp * 0.1): # If we miss more than 10% of failures
    print("   RESULT: FAIL. High False Negative Rate (Dangerous).")
else:
    print("   RESULT: PASS. Safety Critical Logic is sound.")

# CHECK 3: Cross Validation (The "Luck" Test)
# Logic: Train 5 times on different chunks to make sure we didn't just get lucky.
print("\n[Check 3] Cross Validation (Robustness):")
# We use the full dataset here
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"   5-Fold Scores: {cv_scores}")
print(f"   Average Score: {cv_scores.mean():.2%} (+/- {cv_scores.std()*2:.2%})")

print("\n--------------------------")
