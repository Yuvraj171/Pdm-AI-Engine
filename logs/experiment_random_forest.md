# Experiment Log: Random Forest vs XGBoost

**Date:** 2026-01-27
**Status:** Success
**Model:** Random Forest Classifier (`n_estimators=100`, `class_weight='balanced'`)

## 1. Objective

Compare Random Forest performance against the production XGBoost model on the Augmented Data, specifically focusing on "Slow Death" detection (TC-03).

## 2. Key Findings

| Metric | XGBoost (Production) | Random Forest (Experiment) | Winner |
| :--- | :--- | :--- | :--- |
| **Accuracy** | ~86.3% | ~86.2% | Tie |
| **Recall (Failures)** | **~87%**| ~59% (Conservative) | XGBoost |
| **TC-03 Detection** | **PASS** (>90% Risk) | **PASS** (98% Risk) | Tie |
| **Robustness (Gap)** | Healthy | **0.60%** (Extremely Healthy) | Random Forest |
| **Interpretability** | Feature Importance | Feature Importance | Random Forest (Simpler logic) |

## 3. The "Slow Death" Test (TC-03)

- **Scenario**: Safe Pressure (3.2), High Drift (-0.06).
- **Random Forest Result**:
  - Prediction: **FAILURE (1)**
  - Risk Score: **100.00%**
- **Conclusion**: Random Forest successfully learned the rule "Drift > Pressure" thanks to the Augmented Data.

## 4. Optimization Journey

1. **Initial Run**: Recall 0.59, Good Accuracy.
2. **Grid Search**: Found `n_estimators=200`, but Overfitted (Gap 9%).
3. **Manual Regularization**: `max_depth=12`, `min_samples_leaf=10`.
    - **Result**: Cured Overfitting (Gap 0.60%).
    - **Trade-off**: Recall stayed conservative (0.59).

## 5. Final Recommendation

* **Production (Safety Priority)**: Use **XGBoost** (`final_machine_doctor.json`). It catches more failures (87% Recall).
- **Audit/Verification**: Use **Random Forest**. It is extremely stable and resistant to noise. If Random Forest flags an alarm, it is almost certainly real (Precision 0.96).
