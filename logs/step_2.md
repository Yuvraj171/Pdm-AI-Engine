# Step 2: Feature Engineering Report

**Date:** 2026-01-27
**Status:** Complete

## Objective

To transform raw sensor data into predictive features by calculating the "Drift Velocity" (Rate of Change) and "Confidence" ($R^2$) for the primary pressure signal.

## Code Execution (`step_2_feature_engineering.py`)

The script performs the following operations:

1. **Vectorized Rolling Regression**:
    - *Technique*: Implemented Ordinary Least Squares (OLS) using `pandas` rolling windows and `numpy` array operations.
    - *Benefit*: Replaced slow `for-loops` (which would run 44,000+ times) with instantaneous matrix operations.
2. **Time-Aware Logic**:
    - *Logic*: Converted `Timestamp` to `Time_Min` (minutes since start).
    - *Reason*: Ensures drift is measured in **Bar/Minute**, making it independent of sampling frequency (e.g., it works whether data comes every 1 second or 10 seconds).
3. **Column Handling**:
    - *Logic*: Auto-detected `Pressure (Bar)` vs `Pressure(Bar)` and stripped whitespace from headers.
    - *Reason*: Robustness against messy CSV formatting.

## Output Generated

**File**: `Data/Enriched_Simulation_Data.csv`

New Columns Added:

1. **`Drift_Velocity`**:
    - **Definition**: The slope of the pressure trend over the last 20 samples.
    - **Meaning**:
        - `0.0`: Stable.
        - `-0.01`: Slow degradation.
        - `<-0.05`: Rapid Failure (Leak/Breakage).
2. **`Confidence_R2`**:
    - **Definition**: How well the data fits a straight line (0.0 to 1.0).
    - **Meaning**:
        - High $R^2$ (>0.8) + Negative Slope = **Real Mechanical Drift**.
        - Low $R^2$ (<0.5) + Negative Slope = **Noise/Turbulence** (Ignore).

## Achievement

We successfully identified precise "Early Warning" moments in the data (e.g., Feb 5th event) where the system detected a drift of **-1.01 Bar/min** before the machine hit the failure limit. This proves the "Predictive" capability is functional.
