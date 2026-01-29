import pandas as pd
import os

# Define the file path
file_path = os.path.join('Data', 'Simulation_Report_2026-01-27 (6).csv')

# 1. Load the data
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    exit(1)

df = pd.read_csv(file_path)

# 2. Clean 'Shift' names (Fixing 'Shift A' vs 'Shift_A')
if 'Shift' in df.columns:
    df['Shift'] = df['Shift'].str.replace(' ', '_')

# 3. Convert Timestamp to real date/time objects
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp')

# 4. Convert 'Is Anomaly' to numbers (0 and 1)
# Note: Use this if your column has True/False or YES/NO
if 'Is Anomaly' in df.columns:
    # Check unique values first to be safe
    print(f"Unique values in 'Is Anomaly': {df['Is Anomaly'].unique()}")
    # Assuming standard mapping, but handling potential variations
    df['Is Anomaly'] = df['Is Anomaly'].apply(lambda x: 1 if str(x).upper() in ['YES', 'TRUE', '1'] else 0)

# 5. Check for missing values
print("Missing values in each column:\n", df.isnull().sum())

# Save the cleaned data for the next step
cleaned_file_path = os.path.join('Data', 'cleaned_simulation_data.csv')
df.to_csv(cleaned_file_path, index=False)
print(f"\nStep 1 Complete. Data is cleaned and sorted. Saved to {cleaned_file_path}")
