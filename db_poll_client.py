import time
import requests
import sqlite3
import json
import numpy as np
from collections import deque
from scipy.stats import linregress

# CONFIGURATION
DB_PATH = "simulator.db" # UPDATE THIS
API_URL = "http://localhost:8000/predict"
POLL_INTERVAL = 0.5 
WINDOW_SIZE = 20 # Same as training

class FeatureEngineer:
    def __init__(self):
        self.history = deque(maxlen=WINDOW_SIZE)
        
    def add_reading(self, timestamp_sec, pressure):
        self.history.append((timestamp_sec, pressure))
        
    def calculate_features(self):
        # Need enough data to calculate drift
        if len(self.history) < 5:
            return 0.0, 1.0 # Default Safe
            
        data = np.array(self.history)
        t = data[:, 0]
        y = data[:, 1]
        
        # Normalize time to minutes (relative to start of window to keep numbers small)
        t_rel = (t - t[0]) / 60.0
        
        # Variance Check (The "Flatline" Fix)
        if np.var(y) < 0.0001:
            return 0.0, 1.0 # Perfect Stability
            
        try:
            slope, intercept, r_value, p_value, std_err = linregress(t_rel, y)
            r2 = r_value**2
            return slope, r2
        except:
            return 0.0, 0.0 # Math error safe fallback

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def poll_and_process():
    print("🕵️ Smart Connector (with Feature Engineering) Started...")
    
    engineer = FeatureEngineer()
    last_processed_id = 0
    
    # 1. WARM UP (Backfill History)
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        # Fetch last 20 rows to fill buffer
        cur.execute("SELECT timestamp, pressure FROM sensor_logs ORDER BY id DESC LIMIT ?", (WINDOW_SIZE,))
        rows = sorted(cur.fetchall(), key=lambda r: r['timestamp']) # Order by time ASC
        for r in rows:
            # parsing timestamp might be needed depending on DB format
            # ensure 'timestamp' is in seconds (unix epoch)
            ts = r['timestamp'] 
            engineer.add_reading(ts, r['pressure'])
        print(f"   Buffer Warmed with {len(rows)} points.")
        conn.close()
    except Exception as e:
        print(f"   Warmup Skipped: {e}")

    # 2. MAIN LOOP
    while True:
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            
            # Fetch NEW rows
            cur.execute("""
                SELECT id, timestamp, pressure, temp, scan_speed, machine_state 
                FROM sensor_logs 
                WHERE id > ? 
                ORDER BY id ASC LIMIT 50
            """, (last_processed_id,))
            
            rows = cur.fetchall()
            
            if not rows:
                time.sleep(POLL_INTERVAL)
                conn.close()
                continue
                
            for row in rows:
                # Update History
                ts = row['timestamp'] # Ensure this is float seconds
                engineer.add_reading(ts, row['pressure'])
                
                # Calculate Derived Features
                drift, r2 = engineer.calculate_features()
                
                # Prepare Payload
                payload = {
                    "pressure": row['pressure'],
                    "drift": drift,  # Calculated Locally!
                    "r2": r2,        # Calculated Locally!
                    "temp": row['temp'],
                    "scan_speed": row['scan_speed'],
                    "machine_state": row['machine_state']
                }
                
                # Send to AI
                try:
                    response = requests.post(API_URL, json=payload, timeout=0.1)
                    if response.status_code == 200:
                        result = response.json()
                        risk = result['risk_score']
                        status = result['status']
                        
                        # Write Back
                        cur.execute("""
                            UPDATE sensor_logs 
                            SET ai_risk_score = ?, ai_status = ? 
                            WHERE id = ?
                        """, (risk, status, row['id']))
                        print(f"✅ ID {row['id']}: Drift={drift:.4f} --> {status}")
                        
                except Exception as api_err:
                    print(f"⚠️ API Miss: {api_err}")
                
                last_processed_id = row['id']
                conn.commit() # Commit batch frequently
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Loop Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    poll_and_process()
