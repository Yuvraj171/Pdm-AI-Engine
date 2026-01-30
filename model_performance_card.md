# 🧠 AI Machine Doctor: Model Performance Card

## 1. Model Overview

- **Model Type:** XGBoost Classifier + Random Forest (Dual Brain)
- **Purpose:** Detect "Silent Failures" (Hydraulic Leaks, Flow Issues) *before* they cause defects.
- **Training Date:** 2026-01-29 (Updated)
- **Trained On:** Hybrid Dataset (Real Simulation Data + Synthetic "Vaccine" Injection)

---

## 2. Input Features (6 Total)

| Feature | Unit | OK Range | NG Range | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Pressure** | Bar | 3.4 - 3.6 | < 3.0 | Raw hydraulic force. |
| **Drift Velocity** | Bar/sec | -0.01 to +0.01 | < -0.05 | **KEY.** Slope of pressure over time. |
| **Confidence R²** | 0.0-1.0 | > 0.8 | < 0.5 | Signal quality metric. |
| **Part Temp** | °C | **830 - 870** | < 830 \| > 870 | User Spec. Metal Temp. |
| **Scan Speed** | mm/s | 9.0 - 11.0 | < 5.0 | Induction coil speed. |
| **Quench Flow** | LPM | **80 - 150** | 50-80 \| > 150 | User Spec. Coolant Flow. |

---

## 3. Training Scenarios

### Scenario A: "Golden Run" (PASS)

- All parameters within OK range.
- Label: `Is Anomaly = 0`

### Scenario B: "Slow Death" - Drift Leak (FAIL)

- Pressure looks safe (3.2), but Drift is negative (-0.06).
- Label: `Is Anomaly = 1`

### Scenario C: "Flow Failure" (FAIL)

- Temp/Pressure OK, but Quench Flow < 50 (Pump Failure) or > 150 (Flood).
- Label: `Is Anomaly = 1`

---

## 4. How to Retrain

### Step 1: Generate Training Data (Run XGBoost Script)

```powershell
python step_5_final_model.py
```

This creates `Data/Augmented_Training_Data.csv` and `final_machine_doctor.json`.

### Step 2: Train Random Forest (Separate Script)

```powershell
python step_4_train_random_forest.py
```

This loads the augmented data and saves `final_random_forest.joblib`.

### Step 3: Restart API

```powershell
# In Terminal 1:
ctrl+c
python ai_api.py
```

---

## 5. Performance Metrics

- **Accuracy:** ~99% (on Validation Set)
- **False Positive Rate:** < 1%
- **Inference Speed:** ~50ms
