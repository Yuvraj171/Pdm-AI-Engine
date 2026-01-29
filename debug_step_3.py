import pandas as pd
from sklearn.model_selection import train_test_split

print("--- DIAGNOSTIC START ---")
# 1. Load Data
try:
    df = pd.read_csv('Data/Enriched_Simulation_Data.csv')
    print(f"Loaded {len(df)} rows.")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# 2. Split (Identical logic to Step 3)
features = ['Pressure(Bar)', 'Drift_Velocity', 'Confidence_R2', 'Quench Temp(C)', 'Scan Speed']
X = df[features]
y = df['Is Anomaly']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 3. Analyze Distribution
print(f"\nTraining Set ({len(y_train)} rows):")
print(y_train.value_counts(normalize=True))
print(f"Anomalies in Train: {y_train.sum()}")

print(f"\nTesting Set ({len(y_test)} rows):")
print(y_test.value_counts(normalize=True))
print(f"Anomalies in Test: {y_test.sum()}")

print("\n--- DIAGNOSTIC END ---")
