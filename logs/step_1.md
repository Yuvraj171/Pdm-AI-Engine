# Step 1: Data Sanitization Report

**Date:** 2026-01-27
**Status:** Complete

## Objective

To prepare the raw simulation data (`Simulation_Report_2026-01-27 (6).csv`) for Machine Learning by resolving inconsistencies, formatting timestamps, and encoding categorical variables.

## Code Execution (`step_1_data_sanitization.py`)

The script performs the following operations:

1. **Load Data**: Reads the raw CSV file from the `Data` directory.
2. **Standardize 'Shift' Column**:
    - *Logic*: `df['Shift'] = df['Shift'].str.replace(' ', '_')`
    - *Reason*: To treat "Shift A" and "Shift_A" as the same entity.
3. **Timestamp Conversion**:
    - *Logic*: `pd.to_datetime(df['Timestamp'])` and `sort_values('Timestamp')`
    - *Reason*: ML models for drift detection require strictly sequential time-series data.
4. **Target Encoding ('Is Anomaly')**:
    - *Logic*: Converts "YES"/"TRUE" to `1` and "NO"/"FALSE" to `0`.
    - *Reason*: Machine Learning algorithms require numerical input for the target variable.
5. **Missing Value Check**:
    - *Logic*: `df.isnull().sum()`
    - *Reason*: To identify if imputation or row dropping is necessary (currently just reporting).
6. **Save Output**:
    - Saves the processed dataframe to `Data/cleaned_simulation_data.csv`.

## Changes Applied to Data

| Feature | Original State | Transformed State |
| :--- | :--- | :--- |
| **Shift** | Inconsistent (e.g., "Shift A", "Shift_A") | Standardized (e.g., "Shift_A") |
| **Timestamp** | String content | Datetime objects, Sorted chronologically |
| **Is Anomaly** | Categorical ("YES"/"NO") | Binary (1/0) |
| **File Structure** | Raw CSV | Cleaned CSV ready for Feature Engineering |

## Outcome

- **Input File**: `Data/Simulation_Report_2026-01-27 (6).csv`
- **Output File**: `Data/cleaned_simulation_data.csv`
- **Next Step**: Feature Engineering (Drift Velocity Calculation).
