from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from xgboost import XGBClassifier
import numpy as np

# --- 1. INITIALIZATION ---
app = FastAPI(title="AI Sentinel Core", description="Microservice for Early Downtime Detection")

print("🔌 AI MICROSERVICE STARTING...")

# Load Models (Global Scope to keep them in memory)
try:
    print("   [1/2] Loading XGBoost (Paramedic)...")
    xgb_model = XGBClassifier()
    xgb_model.load_model('final_machine_doctor.json')
    
    print("   [2/2] Loading Random Forest (Surgeon)...")
    rf_model = joblib.load('final_random_forest.joblib')
    
    print("✅ SYSTEM READY. Models Loaded.")
except Exception as e:
    print(f"❌ FATAL ERROR: Could not load models. {e}")
    xgb_model = None
    rf_model = None

# --- 2. DATA STRUCTURE ---
class SensorData(BaseModel):
    pressure: float
    drift: float
    r2: float
    temp: float
    scan_speed: float
    flow: float = 120.0  # Quench Flow in LPM (Default: Normal)
    machine_state: str = "UNKNOWN" # e.g. "HEATING", "QUENCH", "LOADING"

# --- 3. LOGIC KERNEL ---
@app.post("/predict")
def predict_health(data: SensorData):
    """
    Main Inference Endpoint.
    Receives Sensor Data -> Applies Guardrails -> Returns Health Prediction.
    """
    
    # DEBUG LOGGING (Temporary)
    print(f"DEBUG: State='{data.machine_state}', Pressure={data.pressure}, Flow={data.flow}")

    # --- GUARDRAIL 1: SAFETY INTERLOCKS (Context Awareness) ---
    # Rule: Only run AI during 'QUENCH' or 'COMPLETED' (Simulator quirk). 
    # Ignore HEATING/LOADING (unless pressure is high).
    valid_states = ["QUENCH", "COMPLETED"]
    # Normalize state: Upper case and strip whitespace
    current_state = data.machine_state.upper().strip()
    
    if current_state not in valid_states:
        # Secondary Check: If pressure is high (>1.0), assume running regardless of state label
        if data.pressure < 1.0:
            return {
                "status": "STANDBY",
                "risk_score": 0.0,
                "message": f"AI Paused (State: {data.machine_state})",
                "confidence": 0.0
            }

    # Rule: Warm-up Check. If Pressure is near zero, machine is off.
    if data.pressure < 0.5:
        return {
            "status": "OFFLINE",
            "risk_score": 0.99, # Technicially high risk if running, but we mark offline
            "message": "Sensor Signal Low / Machine Off",
            "confidence": 0.0
        }

    # --- GUARDRAIL 2: NOISE FILTERING (The 'Deadband') ---
    # Force Drift to 0 if it's just sensor jitter
    features = {
        'Pressure(Bar)': data.pressure,
        'Drift_Velocity': 0.0 if abs(data.drift) < 0.005 else data.drift,
        'Confidence_R2': data.r2,
        'Part Temp(C)': data.temp,
        'Scan Speed': data.scan_speed,
        'Quench Flow(LPM)': data.flow
    }
    
    # Prepare DataFrame
    input_df = pd.DataFrame([features])[['Pressure(Bar)', 'Drift_Velocity', 'Confidence_R2', 'Part Temp(C)', 'Scan Speed', 'Quench Flow(LPM)']]

    # --- INFERENCE: THE "DOUBLE DOCTOR" ---
    
    # Step A: XGBoost (Fast Check)
    try:
        xgb_prob = float(xgb_model.predict_proba(input_df)[0][1])
        xgb_pred = int(xgb_model.predict(input_df)[0])
    except Exception as e:
        return {"error": f"Inference Failed: {e}"}

    # --- NEW: GRADUAL DRIFT-BASED RISK CALCULATION ---
    # This creates a smooth risk curve based on drift velocity magnitude
    # Drift thresholds: -0.01 = safe, -0.05 = warning, -0.1 = critical
    drift_val = features['Drift_Velocity']
    
    # Calculate drift-based risk (0.0 to 1.0)
    if drift_val >= -0.01:
        drift_risk = 0.0  # No significant drift
    elif drift_val >= -0.05:
        # Linear interpolation: -0.01 → 0%, -0.05 → 50% (Warning zone)
        drift_risk = (abs(drift_val) - 0.01) / 0.04 * 0.5
    elif drift_val >= -0.1:
        # Linear interpolation: -0.05 → 50%, -0.1 → 100% (Critical zone)
        drift_risk = 0.5 + (abs(drift_val) - 0.05) / 0.05 * 0.5
    else:
        drift_risk = 1.0  # Maximum risk
    
    # Blend ML probability with drift-based risk (weighted average)
    # This ensures gradual progression even when ML model is confident
    blended_risk = max(xgb_prob, drift_risk * 0.8 + xgb_prob * 0.2)
    
    # Step B: Determine Status based on BLENDED risk
    if blended_risk < 0.4:
        return {
            "status": "OPTIMAL",
            "risk_score": blended_risk,
            "message": "System Nominal",
            "rca": "NONE",
            "drift_velocity": drift_val
        }
    elif blended_risk < 0.8:
        return {
            "status": "WARNING",
            "risk_score": blended_risk,
            "message": f"Drift Detected: {drift_val:.4f} bar/min",
            "rca": "EARLY_DRIFT",
            "drift_velocity": drift_val
        }
    else:
        # Verify with Random Forest for critical cases
        rf_prob = float(rf_model.predict_proba(input_df)[0][1])
        rf_pred = int(rf_model.predict(input_df)[0])
        
        if rf_pred == 1 or blended_risk > 0.9:
            return {
                "status": "CRITICAL_FAILURE",
                "risk_score": max(blended_risk, rf_prob),
                "message": "EMERGENCY: Drift Confirmed",
                "rca": "DRIFT_CONFIRMED",
                "drift_velocity": drift_val
            }
        else:
            return {
                "status": "WARNING",
                "risk_score": blended_risk,
                "message": "High Drift - Monitoring",
                "rca": "EARLY_DRIFT",
                "drift_velocity": drift_val
            }

@app.get("/health")
def api_health_check():
    return {"status": "online", "models_loaded": xgb_model is not None}

if __name__ == "__main__":
    import uvicorn
    print("🚀 STARTING API SERVER on Port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8100)
