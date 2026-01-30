import time
import requests
import sqlite3
import json
import numpy as np
from collections import deque
from scipy.stats import linregress
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================
DB_PATH = r"D:\New folder (2)\Machine-Simulator\backend\simulation_v2.db"
API_URL = "http://localhost:8100/predict"
POLL_INTERVAL = 0.5  # seconds between polls
WINDOW_SIZE = 20     # Rolling window for feature engineering (same as training)

# =============================================================================
# COLUMN MAPPING (Telemetry Table → AI Model)
# =============================================================================
# Telemetry columns:    quench_pressure, quench_water_temp, coil_scan_speed, state
# AI Model expects:     pressure,        temp,              scan_speed,       machine_state

class FeatureEngineer:
    """
    Calculates derived features (Drift_Velocity, Confidence_R2) in real-time
    using a rolling window of pressure readings.
    """
    def __init__(self):
        self.history = deque(maxlen=WINDOW_SIZE)
        
    def add_reading(self, timestamp_sec: float, pressure: float):
        """Add a new pressure reading to the rolling buffer."""
        self.history.append((timestamp_sec, pressure))
        
    def calculate_features(self) -> tuple[float, float]:
        """
        Calculate Drift_Velocity and Confidence_R2 from the rolling window.
        Returns (drift, r2) tuple.
        """
        # Need enough data to calculate drift
        if len(self.history) < 5:
            return 0.0, 1.0  # Default: Safe, stable
            
        data = np.array(self.history)
        t = data[:, 0]  # timestamps
        y = data[:, 1]  # pressure values
        
        # Normalize time to minutes (relative to start of window)
        t_rel = (t - t[0]) / 60.0
        
        # Variance Check (The "Flatline" Fix)
        # If pressure is perfectly stable, no drift
        if np.var(y) < 0.0001:
            return 0.0, 1.0  # Perfect Stability
            
        try:
            slope, intercept, r_value, p_value, std_err = linregress(t_rel, y)
            r2 = r_value ** 2
            return slope, r2
        except:
            return 0.0, 0.0  # Math error fallback


def parse_timestamp(ts_value) -> float:
    """
    Convert timestamp to Unix epoch seconds.
    Handles both float (already epoch) and datetime string formats.
    """
    if isinstance(ts_value, (int, float)):
        return float(ts_value)
    
    # Try common datetime formats
    formats = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(str(ts_value), fmt)
            return dt.timestamp()
        except ValueError:
            continue
    
    # Fallback: current time
    return time.time()


def get_db_connection():
    """Create a connection to the simulator database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def poll_and_process():
    """
    Main polling loop.
    1. Warms up the feature engineer with historical data
    2. Polls for new telemetry rows
    3. Calculates drift features
    4. Sends to AI API
    5. Writes results back to DB
    """
    print("=" * 60)
    print("🕵️  SMART CONNECTOR v2.0")
    print("   Real-Time Feature Engineering + AI Integration")
    print("=" * 60)
    print(f"   DB Path:  {DB_PATH}")
    print(f"   API URL:  {API_URL}")
    print(f"   Window:   {WINDOW_SIZE} points")
    print("=" * 60)
    
    engineer = FeatureEngineer()
    last_processed_id = 0
    
    # -------------------------------------------------------------------------
    # 1. WARM UP (Backfill History from existing data)
    # -------------------------------------------------------------------------
    try:
        print("⏳ Checking database state...")
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Find the last row that actually has an AI prediction
        cur.execute("SELECT MAX(id) FROM telemetry WHERE ai_status IS NOT NULL")
        row = cur.fetchone()
        last_processed_id = row[0] if row and row[0] else 0
        
        # CHECK FOR HUGE LAG (If > 1000 rows pending, skip to end for Live Demo)
        cur.execute("SELECT MAX(id) FROM telemetry")
        max_id = cur.fetchone()[0] or 0
        
        if (max_id - last_processed_id) > 1000:
            print(f"⏩ HUGE LAG DETECTED ({max_id - last_processed_id} rows). Skipping history to go LIVE...")
            last_processed_id = max_id
        
        print(f"   Last Analyzed ID: {last_processed_id}")
        
        # Warm up history based on that point
        cur.execute("""
            SELECT timestamp_sim, quench_pressure 
            FROM telemetry 
            WHERE id <= ? 
            ORDER BY id DESC LIMIT ?
        """, (last_processed_id, WINDOW_SIZE))
        
        warmup_rows = cur.fetchall()
        
        if warmup_rows:
            for r in reversed(warmup_rows):
                ts = parse_timestamp(r['timestamp_sim'])
                pressure = r['quench_pressure'] or 0.0 # Ensure pressure is float
                engineer.add_reading(ts, pressure)
            print(f"✅ Buffer Warmed with {len(warmup_rows)} points.")
        else:
            print("⚠️  No history found, starting fresh.")
            
        conn.close()
    except Exception as e:
        print(f"⚠️  Warmup Skipped: {e}")

    print("\n🔄 Entering Main Loop... (Ctrl+C to stop)\n")
    
    # -------------------------------------------------------------------------
    # 2. MAIN LOOP
    # -------------------------------------------------------------------------
    while True:
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            
            # Fetch NEW rows from telemetry table
            cur.execute("""
                SELECT id, timestamp_sim, 
                       quench_pressure, quench_water_temp, coil_scan_speed, state, part_temp, quench_water_flow
                FROM telemetry 
                WHERE id > ? 
                ORDER BY id ASC 
                LIMIT 50
            """, (last_processed_id,))
            
            rows = cur.fetchall()
            
            if not rows:
                time.sleep(POLL_INTERVAL)
                conn.close()
                continue
                
            for row in rows:
                # ----------------------------------------------------------
                # A. Update Feature Engineer History
                # ----------------------------------------------------------
                ts = parse_timestamp(row['timestamp_sim'])
                pressure = row['quench_pressure'] or 0.0
                engineer.add_reading(ts, pressure)
                # ----------------------------------------------------------
                # B. Calculate Derived Features
                # ----------------------------------------------------------
                drift, r2 = engineer.calculate_features()
                
                # ----------------------------------------------------------
                # B2. Write drift/r2 TO DB (for visibility & debugging)
                # ----------------------------------------------------------
                try:
                    cur.execute("""
                        UPDATE telemetry 
                        SET drift_velocity = ?, confidence_r2 = ?
                        WHERE id = ?
                    """, (drift, r2, row['id']))
                except sqlite3.OperationalError as col_err:
                    if "no such column" in str(col_err):
                        # First run - add columns
                        print("📊 Adding drift_velocity & confidence_r2 columns to DB...")
                        cur.execute("ALTER TABLE telemetry ADD COLUMN drift_velocity REAL DEFAULT 0")
                        cur.execute("ALTER TABLE telemetry ADD COLUMN confidence_r2 REAL DEFAULT 0")
                        cur.execute("""
                            UPDATE telemetry 
                            SET drift_velocity = ?, confidence_r2 = ?
                            WHERE id = ?
                        """, (drift, r2, row['id']))
                
                # ----------------------------------------------------------
                # C. Prepare Payload (Map telemetry → AI model columns)
                # ----------------------------------------------------------
                payload = {
                    "pressure": pressure,
                    "drift": drift,                          # Calculated locally!
                    "r2": r2,                                # Calculated locally!
                    "temp": row['part_temp'] or 850.0,       # FIXED: Model expects Metal Temp (~850C), not Water
                    "scan_speed": row['coil_scan_speed'] or 10.0,
                    "flow": row['quench_water_flow'] or 120.0, # NEW: Quench Flow (LPM)
                    "machine_state": row['state'] or "UNKNOWN"
                }
                
                # ----------------------------------------------------------
                # D. Send to AI API
                # ----------------------------------------------------------
                try:
                    response = requests.post(API_URL, json=payload, timeout=0.5)
                    
                    if response.status_code == 200:
                        result = response.json()
                        risk = result.get('risk_score', 0.0)
                        status = result.get('status', 'UNKNOWN')
                        
                        # --------------------------------------------------
                        # E. Write Results Back to Database (UNIFIED UPDATE)
                        # --------------------------------------------------
                        try:
                            cur.execute("""
                                UPDATE telemetry 
                                SET ai_risk_score = ?, ai_status = ?,
                                    drift_velocity = ?, confidence_r2 = ?
                                WHERE id = ?
                            """, (risk, status, drift, r2, row['id']))
                        except sqlite3.OperationalError as db_err:
                            # Columns might not exist yet - try minimal update
                            if "no such column" in str(db_err):
                                print(f"⚠️  DB Schema: Adding missing columns...")
                                try:
                                    cur.execute("ALTER TABLE telemetry ADD COLUMN ai_risk_score REAL DEFAULT 0")
                                except: pass
                                try:
                                    cur.execute("ALTER TABLE telemetry ADD COLUMN ai_status TEXT DEFAULT 'UNKNOWN'")
                                except: pass
                                try:
                                    cur.execute("ALTER TABLE telemetry ADD COLUMN drift_velocity REAL DEFAULT 0")
                                except: pass
                                try:
                                    cur.execute("ALTER TABLE telemetry ADD COLUMN confidence_r2 REAL DEFAULT 0")
                                except: pass
                        
                        # Status emoji based on risk
                        emoji = "🟢" if risk < 0.3 else ("🟡" if risk < 0.7 else "🔴")
                        print(f"{emoji} ID {row['id']:>6}: Drift={drift:+.4f} | R²={r2:.2f} → {status} ({risk:.1%})")
                        
                        if status == 'STANDBY' and result.get('message'):
                             print(f"      📝 Reason: {result['message']}")
                        
                except requests.exceptions.Timeout:
                    print(f"⏱️  ID {row['id']}: API timeout")
                except requests.exceptions.ConnectionError:
                    print(f"❌ ID {row['id']}: API not available")
                except Exception as api_err:
                    print(f"⚠️  ID {row['id']}: API Error - {api_err}")
                
                last_processed_id = row['id']
            
            conn.commit()
            conn.close()
            
        except KeyboardInterrupt:
            print("\n\n👋 Shutting down Smart Connector...")
            break
        except Exception as e:
            print(f"❌ Loop Error: {e}")
            time.sleep(1)


if __name__ == "__main__":
    poll_and_process()
