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
    machine_state: str = "UNKNOWN" # e.g. "HEATING", "QUENCH", "LOADING"

# --- 3. LOGIC KERNEL ---
@app.post("/predict")
def predict_health(data: SensorData):
    """
    Main Inference Endpoint.
    Receives Sensor Data -> Applies Guardrails -> Returns Health Prediction.
    """
    
    # --- GUARDRAIL 1: SAFETY INTERLOCKS (Context Awareness) ---
    # Rule: Only run AI during 'QUENCH'. Ignore HEATING/LOADING (Noise).
    if data.machine_state.upper() != "QUENCH":
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
        'Quench Temp(C)': data.temp,
        'Scan Speed': data.scan_speed
    }
    
    # Prepare DataFrame
    input_df = pd.DataFrame([features])[['Pressure(Bar)', 'Drift_Velocity', 'Confidence_R2', 'Quench Temp(C)', 'Scan Speed']]

    # --- INFERENCE: THE "DOUBLE DOCTOR" ---
    
    # Step A: XGBoost (Fast Check)
    try:
        xgb_prob = float(xgb_model.predict_proba(input_df)[0][1])
        xgb_pred = int(xgb_model.predict(input_df)[0])
    except Exception as e:
        return {"error": f"Inference Failed: {e}"}

    # Step B: Logic Tree
    if xgb_pred == 0:
        # XGBoost says Safe
        return {
            "status": "OPTIMAL",
            "risk_score": xgb_prob,
            "message": "System Nominal",
            "rca": "NONE"
        }
    else:
        # XGBoost says DANGER -> Verify with Random Forest
        rf_prob = float(rf_model.predict_proba(input_df)[0][1])
        rf_pred = int(rf_model.predict(input_df)[0])
        
        if rf_pred == 1:
            # Both Agree -> CRITICAL
            return {
                "status": "CRITICAL_FAILURE",
                "risk_score": max(xgb_prob, rf_prob),
                "message": "EMERGENCY: Drift Detected by Dual Models",
                "rca": "DRIFT_CONFIRMED"
            }
        else:
            # Disagreement -> WARNING
            return {
                "status": "WARNING",
                "risk_score": xgb_prob,
                "message": "Potential Anomaly (Model Disagreement)",
                "rca": "EARLY_DRIFT"
            }

@app.get("/health")
def api_health_check():
    return {"status": "online", "models_loaded": xgb_model is not None}
