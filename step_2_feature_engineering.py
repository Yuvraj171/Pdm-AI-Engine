import pandas as pd
import numpy as np
import os

# Define file paths
input_file = os.path.join('Data', 'cleaned_simulation_data.csv')
output_file = os.path.join('Data', 'Enriched_Simulation_Data.csv')

# 1. Load and Prepare Time-Aware Data
if not os.path.exists(input_file):
    print(f"Error: File {input_file} not found.")
    exit(1)

df = pd.read_csv(input_file)
# CLEANUP: Strip whitespace from column names
df.columns = df.columns.str.strip()

df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values('Timestamp').reset_index(drop=True)

# Create X variable: Minutes since start (Float)
df['Time_Min'] = (df['Timestamp'] - df['Timestamp'].min()).dt.total_seconds() / 60.0

# Identify Pressure Column
possible_cols = ['Pressure (Bar)', 'Pressure(Bar)']
pressure_col = next((c for c in possible_cols if c in df.columns), None)
if not pressure_col:
    print(f"Error: Pressure column not found. Available columns: {df.columns.tolist()}")
    exit(1)
print(f"Using pressure column: '{pressure_col}'")

def calculate_rolling_regression_fast(df, col_name, window_size=20):
    """Calculates Slope (Bar/Min) and R2 using vectorized math."""
    n = window_size
    x = df['Time_Min']
    y = df[col_name]
    
    # Calculate rolling components
    s_x = x.rolling(n).sum()
    s_y = y.rolling(n).sum()
    s_xx = (x**2).rolling(n).sum()
    s_yy = (y**2).rolling(n).sum()
    s_xy = (x*y).rolling(n).sum()
    
    # OLS Slope formula: (n*sum_xy - sum_x*sum_y) / (n*sum_xx - sum_x^2)
    numerator = (n * s_xy - s_x * s_y)
    denominator = (n * s_xx - s_x**2)
    
    # Handle division by zero (if x doesn't vary in window, though unlikely with Time_Min)
    slope = np.where(denominator != 0, numerator / denominator, 0)
    
    # R-Squared formula
    # r = (n*sum_xy - sum_x*sum_y) / sqrt((n*sum_xx - sum_x^2) * (n*sum_yy - sum_y^2))
    denom_r_sq = (n * s_xx - s_x**2) * (n * s_yy - s_y**2)
    
    # Protect against negative values in sqrt (floating point errors) and zero division
    valid_r = denom_r_sq > 0
    numerator_r = n * s_xy - s_x * s_y
    
    r_squared = np.zeros_like(slope)
    r_squared[valid_r] = (numerator_r[valid_r] / np.sqrt(denom_r_sq[valid_r]))**2
    
    return slope, r_squared

# 2. Apply the Logic (Instantaneous)
print("Calculating Drift Velocity and R2...")
slope, r2 = calculate_rolling_regression_fast(df, pressure_col)
df['Drift_Velocity'] = slope
df['Confidence_R2'] = r2

# 3. Identify 'Early Warning' Events
# Threshold: Dropping faster than 0.005 Bar/min with 80% confidence
warnings = df[(df['Drift_Velocity'] < -0.005) & (df['Confidence_R2'] > 0.8)]

print(f"Processed {len(df)} rows.")
print(f"Found {len(warnings)} Early Warning points.")
if not warnings.empty:
    print(warnings[['Timestamp', pressure_col, 'Drift_Velocity', 'Confidence_R2']].head())

# Save for the next step
df.to_csv(output_file, index=False)
print(f"Step 2 Complete. Enriched data saved to {output_file}")
