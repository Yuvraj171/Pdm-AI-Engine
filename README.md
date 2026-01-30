# 🧠 PDM AI Engine - Predictive Maintenance System

A real-time **AI-powered early warning system** that detects machine failures **before they happen** by monitoring sensor drift patterns.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

---

## 🎯 What Does This Project Do?

This system monitors industrial machine telemetry (pressure, temperature, flow) and uses **dual AI models** to detect anomalies that precede breakdowns.

| Traditional Approach         | This AI System                 |
|------------------------------|--------------------------------|
| React AFTER breakdown        | Predict 2-5 min BEFORE         |
| Downtime = lost production   | Early warning = time to fix    |
| "Machine stopped!"           | "Machine WILL stop in 3 min"   |

### Key Capability: **Drift Detection**

The AI doesn't just look at current values — it analyzes the **trend (slope)** over time to catch "silent failures" like slow hydraulic leaks.

```text
Example: Pressure = 3.4 Bar (looks OK!)
         But Drift = -0.06 Bar/min (dropping fast!)
         → AI says: "WARNING: Failure in 2 minutes"
```

---

## 🏗️ Architecture

```text
┌─────────────────────┐
│   Machine Simulator │  (Generates telemetry data)
│   (simulation_v2.db)│
└──────────┬──────────┘
           │ SQLite
           ▼
┌─────────────────────┐      HTTP POST       ┌─────────────────────┐
│   db_poll_client.py │ ──────────────────▶  │     ai_api.py       │
│   (Smart Connector) │                      │   (AI Microservice) │
│   - Reads telemetry │                      │   - XGBoost Model   │
│   - Calculates drift│  ◀──────────────────  │   - Random Forest   │
│   - Writes AI scores│      JSON Response   │   - Risk Scoring    │
└──────────┬──────────┘                      └─────────────────────┘
           │
           ▼
┌─────────────────────┐
│   ai_dashboard.py   │  (Web UI on port 8080)
│   - Live risk chart │
│   - Shift reports   │
│   - Root Cause Anal │
└─────────────────────┘
```

---

## 📁 Project Structure

```text
Pdm-AI-Engine/
├── ai_api.py                 # FastAPI microservice (the AI brain)
├── ai_dashboard.py           # Web dashboard with live charts
├── db_poll_client.py         # Polls DB, calculates drift, calls AI
│
├── step_1_data_sanitization.py   # Clean raw data
├── step_2_feature_engineering.py # Calculate drift features
├── step_3_train_model.py         # Initial model training
├── step_4_stress_test.py         # Validate model accuracy
├── step_5_final_model.py         # Final XGBoost training
├── train_random_forest.py        # Random Forest (second opinion)
│
├── final_machine_doctor.json     # Trained XGBoost model
├── final_random_forest.joblib    # Trained Random Forest model
├── model_performance_card.md     # Model documentation
│
├── Data/
│   ├── Enriched_Simulation_Data.csv    # Real simulation data
│   └── Augmented_Training_Data.csv     # Training data + synthetic
│
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the System (3 terminals)

```powershell
# Terminal 1: AI Microservice (must start first)
python ai_api.py

# Terminal 2: Web Dashboard
python ai_dashboard.py

# Terminal 3: Database Connector
python db_poll_client.py
```

### 3. Open Dashboard

Navigate to: **[http://localhost:8080](http://localhost:8080)**

---

## 🤖 How the AI Works

### Dual Model System ("Double Doctor")

| Model             | Role                           | Speed  |
|-------------------|--------------------------------|--------|
| **XGBoost**       | Fast initial check             | ~5ms   |
| **Random Forest** | Second opinion for edge cases  | ~20ms  |

### Risk Score Calculation

The AI outputs a **Risk Score (0-100%)** based on:

1. **ML Model Probability** — What the trained models predict
2. **Drift Velocity** — How fast pressure is changing

```text
Drift Velocity    → Risk Score → Status
> -0.01 Bar/min   →    0-10%   → OPTIMAL (Green)
-0.01 to -0.05    →   10-50%   → WARNING (Yellow)
-0.05 to -0.10    →   50-80%   → WARNING (Yellow)
< -0.10 Bar/min   →   80-100%  → CRITICAL (Red)
```

### Output States

| Status             | Meaning              | Action     |
|--------------------|----------------------|------------|
| `OPTIMAL`          | All systems normal   | Continue   |
| `WARNING`          | Early drift detected | Monitor    |
| `CRITICAL_FAILURE` | Failure imminent     | Stop & Fix |
| `STANDBY`          | Machine not active   | AI paused  |

---

## 📊 Input Features (6 Total)

| Feature        | Unit    | OK Range        | NG Range           |
|----------------|---------|-----------------|--------------------|
| Pressure       | Bar     | 3.4 - 3.6       | < 3.0              |
| Drift Velocity | Bar/min | -0.01 to +0.01  | < -0.05            |
| Confidence R²  | 0-1     | > 0.8           | < 0.5              |
| Part Temp      | °C      | 830 - 870       | < 830 or > 870     |
| Scan Speed     | mm/s    | 9 - 11          | < 5                |
| Quench Flow    | LPM     | 80 - 150        | < 50 or > 150      |

---

## 🔄 Retraining the Model

### Step 1: Generate Training Data

```bash
python step_5_final_model.py
```

This creates:

- `Data/Augmented_Training_Data.csv`
- `final_machine_doctor.json` (XGBoost)

### Step 2: Train Random Forest

```bash
python train_random_forest.py
```

This creates:

- `final_random_forest.joblib`

### Step 3: Restart AI Service

```bash
# Ctrl+C to stop, then:
python ai_api.py
```

---

## 📈 Training Scenarios

The model is trained on these failure patterns:

| Scenario         | Description                             | Label       |
|------------------|-----------------------------------------|-------------|
| **Golden Run**   | All parameters in range                 | Normal (0)  |
| **Slow Death**   | Pressure OK, but drift is negative      | Anomaly (1) |
| **Flow Failure** | Pressure/Temp OK, but flow abnormal     | Anomaly (1) |

---

## 🔌 API Endpoints

### `POST /predict`

Send sensor data, get risk assessment.

**Request:**

```json
{
  "pressure": 3.4,
  "drift": -0.05,
  "r2": 0.95,
  "temp": 850,
  "scan_speed": 10,
  "flow": 120,
  "machine_state": "QUENCH"
}
```

**Response:**

```json
{
  "status": "WARNING",
  "risk_score": 0.57,
  "message": "Drift Detected: -0.0500 bar/min",
  "rca": "EARLY_DRIFT",
  "drift_velocity": -0.05
}
```

### `GET /health`

Check if API is running and models are loaded.

---

## 📊 Dashboard Features

| Feature               | Description                              |
|-----------------------|------------------------------------------|
| **Shift Performance** | A/B/C grade based on % optimal readings  |
| **Drift Trend Chart** | Live visualization of risk over time     |
| **Safety Alerts**     | List of high-risk events (>80%)          |
| **Automated RCA**     | Root cause analysis for any alert        |

---

## 🛠️ Configuration

Edit these paths in the Python files if needed:

```python
# db_poll_client.py / ai_dashboard.py
DB_PATH = r"D:\...\simulation_v2.db"

# ai_api.py runs on port 8100
# ai_dashboard.py runs on port 8080
```

---

## 📋 Requirements

```text
fastapi
uvicorn
pandas
numpy
scipy
scikit-learn
xgboost
joblib
requests
```

---

## 🎯 Use Case

This system is designed for **maintenance teams** to:

- Get early warnings before machine failure
- Reduce unplanned downtime
- Enable proactive repairs
- Track shift performance over time

---

## 📄 License

Apache 2.0 - See [LICENSE](LICENSE)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request
