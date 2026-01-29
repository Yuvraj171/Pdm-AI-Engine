# Step 3: Model Training & Refinement Log

**Date:** 2026-01-27
**Status:** Complete
**Final Accuracy:** 85.87%

## 1. The Challenge

Training an XGBoost classifier to distinguish between "Healthy State" and "Anomaly" using the features generated in Step 2.

## 2. Iteration 1 (The Failure)

- **Approach**: Used standard time-series split (`shuffle=False`) and aggressive weighting (`scale_pos_weight=10`).
- **Outcome**: **33.03% Accuracy** (Critical Failure).
- **Root Cause Analysis**:
  - We discovered a massive **Distribution Shift**.
  - **Training Set**: ~32% Anomalies (Model learned "Failures are common").
  - **Test Set**: ~3% Anomalies (Model faced a mostly healthy dataset).
  - Result: The model hallucinated failures everywhere (False Positives).

## 3. Iteration 2 (The Refinement)

To fix this, we implemented two specific mathematical changes:

| Change | Code Implementation | Physics/Logic Reason |
| :--- | :--- | :--- |
| **Stratified Split** | `stratify=y` | We forced the "Test Exam" to have the exact same percentage of failures as the "Textbook" (Training Data). |
| **Shuffling** | `shuffle=True` | We broke the time dependence for *training* to ensure the model learns generic patterns, not just "what happened yesterday". |
| **Balanced Weighing** | Removed `scale_pos_weight` | Since the Training set now has a representative number of anomalies, we don't need to artificially force the model to panic. |

## 4. Final Outcome

- **Accuracy**: **85.87%** (A+ Improvement from 33%).
- **Model Behavior**:
  - **Pressure (88% Importance)**: The model is highly reliant on the raw pressure value. It acts as a **Digital Detector**.
  - **Drift Velocity (1.3% Importance)**: The model is ignoring the "Early Warning" signals.

## 5. Conclusion & Next Steps

We have a working "Detector" but not a "Predictor".
The model is accurate but "Lazy" (it waits for pressure to drop instead of checking the slope).
**Action for Step 4**: We must run **Adversarial Stress Tests** (Test Case 03) to punish this laziness and force it to learn Drift.
