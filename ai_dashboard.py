"""
AI Dashboard API (Analyst Edition)
==================================
FastAPI backend for the AI Predictive Maintenance Dashboard.
Focus: Historical Analysis, RCA, and Shift Reporting.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from scipy.stats import linregress
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================
DB_PATH = r"D:\New folder (2)\Machine-Simulator\backend\simulation_v2.db"
DASHBOARD_PORT = 8080

app = FastAPI(
    title="AI Drift Sentinel Dashboard",
    description="Real-time Drift Detection & Diagnostics"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db_connection():
    """Create a connection to the simulator database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# =============================================================================
# REST API ENDPOINTS
# =============================================================================

@app.get("/api/stats")
def get_stats():
    """Get current statistics summary."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get latest reading just for context (risk score)
        cur.execute("""
            SELECT id, timestamp_sim, quench_pressure, ai_risk_score, ai_status, state
            FROM telemetry 
            ORDER BY id DESC 
            LIMIT 1
        """)
        latest = cur.fetchone()
        
        if not latest:
            conn.close()
            return {"error": "No data available"}
        
        conn.close()
        
        return {
            "latest": {
                "id": latest['id'],
                "timestamp": str(latest['timestamp_sim']),
                "risk_score": latest['ai_risk_score'],
                "status": latest['ai_status'],
                "state": latest['state']
            }
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/rca/{event_id}")
def get_rca(event_id: int):
    """Automated Root Cause Analysis (Look-back logic)."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # 1. Get the event details
        cur.execute("SELECT * FROM telemetry WHERE id = ?", (event_id,))
        event = cur.fetchone()
        
        if not event:
            return {"error": "Event not found"}
            
        # 2. Get history -60s from event
        cur.execute("""
            SELECT quench_pressure, timestamp_sim
            FROM telemetry 
            WHERE id <= ? AND id > ?
            ORDER BY id ASC
        """, (event_id, event_id - 60))
        
        rows = cur.fetchall()
        
        if len(rows) < 10:
             return {"rca": "Not enough history data for analysis."}

        # 3. Calculate Drift Trajectory (Slope)
        pressures = [r['quench_pressure'] for r in rows]
        times = np.arange(len(pressures))
        slope, _, _, _, _ = linregress(times, pressures)
        
        # 4. Generate Explanation
        explanation = f"🔍 <b>RCA Report for Event #{event_id}</b>\n"
        explanation += f"Time: {event['timestamp_sim']}\n"
        
        # GROUND TRUTH (From Simulator)
        truth = event['ng_reason'] or event['downtime_reason'] or "None"
        explanation += f"📝 <b>Simulator Label:</b> {truth}\n\n"
        
        if slope < -0.01:
            explanation += f"🔴 <b>AI Diagnosis: RAPID PRESSURE LOSS</b>\n"
            explanation += f"   • Gradient: {slope:.4f} bar/sec (Severe Drop)\n"
            explanation += f"   • Likely Cause: Hydraulic Leak / Valve Failure\n"
        elif slope > 0.01:
            explanation += f"🟠 <b>AI Diagnosis: PRESSURE BUILDUP</b>\n"
            explanation += f"   • Gradient: +{slope:.4f} bar/sec\n"
            explanation += f"   • Likely Cause: Filter Clog / Regulator Stuck\n"
        else:
            explanation += f"⚠️ <b>AI Diagnosis: SUDDEN FAULT (Non-Drift)</b>\n"
            explanation += f"   • Gradient: Stable ({slope:.4f})\n"
            explanation += f"   • Likely Cause: Electrical Fault / E-Stop / Sensor Glitch\n"
            
        return {
            "event_id": event_id,
            "slope": slope,
            "explanation": explanation,
            "timestamp": str(event['timestamp_sim'])
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/api/reports/shift")
def get_shift_report():
    """Shift Performance Report."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("SELECT COUNT(*) FROM telemetry")
        total_rows = cur.fetchone()[0]
        
        if total_rows == 0:
             return {"error": "No data"}

        # First try ai_status based counts
        cur.execute("SELECT ai_status, COUNT(*) FROM telemetry WHERE ai_status IS NOT NULL GROUP BY ai_status")
        counts = {r[0]: r[1] for r in cur.fetchall()}
        
        optimal = counts.get('OPTIMAL', 0)
        warning = counts.get('WARNING', 0)
        critical = counts.get('CRITICAL_FAILURE', 0)
        
        # Calculate TOTAL ANALYZED (Exclude STANDBY/OFFLINE)
        total_analyzed = optimal + warning + critical
        
        # FALLBACK: If AI hasn't processed, use machine state as proxy
        if total_analyzed == 0:
            cur.execute("SELECT state, COUNT(*) FROM telemetry GROUP BY state")
            state_counts = {r[0]: r[1] for r in cur.fetchall()}
            
            # Map states to health: RUNNING/IDLE = optimal, HEATING/QUENCH/LOADING = warning, DOWN = critical
            optimal = state_counts.get('RUNNING', 0) + state_counts.get('IDLE', 0)
            warning = state_counts.get('HEATING', 0) + state_counts.get('QUENCHING', 0) + state_counts.get('LOADING', 0) + state_counts.get('UNLOADING', 0)
            critical = state_counts.get('DOWN', 0) + state_counts.get('BREAKDOWN', 0)
            total_analyzed = optimal + warning + critical
        
        if total_analyzed > 0:
            optimal_pct = (optimal / total_analyzed) * 100
            warning_pct = (warning / total_analyzed) * 100
            critical_pct = (critical / total_analyzed) * 100
        else:
            optimal_pct = 0.0
            warning_pct = 0.0
            critical_pct = 0.0
        
        # Score Logic
        score = "A" if optimal_pct > 95 else "B" if optimal_pct > 80 else "C"
        if total_analyzed == 0: score = "N/A" # No data processed yet
            
        color = "#00ff88" if score == "A" else "#ffcc00" if score == "B" else "#ff4444"
        if score == "N/A": color = "#666"
        
        conn.close()
        
        return {
            "shift_score": score,
            "score_color": color,
            "total_parts": total_rows,
            "health_stats": {
                "optimal": f"{optimal_pct:.1f}%",
                "warning": f"{warning_pct:.1f}%",
                "critical": f"{critical_pct:.1f}%"
            },
            "raw_counts": {
                "optimal": optimal,
                "warning": warning,
                "critical": critical
            },
            "summary": f"Shift is operating at <b style='color:{color}'>Grade {score}</b> stability."
        }
        
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/audit/safety_net")
def get_safety_audit():
    """Safety Net Audit: Find missed failures."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT id, timestamp_sim, ai_risk_score, state, ai_status
            FROM telemetry
            WHERE ai_risk_score > 0.8 
              AND state != 'DOWN'
            ORDER BY id DESC
            LIMIT 50
        """)
        
        rows = cur.fetchall()
        
        misses = []
        for r in rows:
            misses.append({
                "id": r['id'],
                "timestamp": str(r['timestamp_sim']),
                "risk": r['ai_risk_score'],
                "state": r['state'],
                "msg": f"Risk {r['ai_risk_score']*100:.0f}% ignored while {r['state']}"
            })
            
        return {"audit_log": misses}

    except Exception as e:
        return {"error": str(e)}


@app.get("/api/trend")
def get_trend():
    """Get drift trend data."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT id, timestamp_sim, ai_risk_score, quench_pressure
            FROM telemetry 
            WHERE ai_risk_score IS NOT NULL
            ORDER BY id DESC 
            LIMIT 300
        """)
        
        rows = cur.fetchall()
        conn.close()
        
        return {
            "trend": [
                {
                    "id": r['id'],
                    "timestamp": str(r['timestamp_sim']),
                    "risk": r['ai_risk_score'],
                    "pressure": r['quench_pressure']
                }
                for r in reversed(rows)
            ]
        }
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# WEBSOCKET
# =============================================================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    last_id = 0
    
    try:
        while True:
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                
                cur.execute("""
                    SELECT id, timestamp_sim, ai_risk_score, ai_status, quench_pressure
                    FROM telemetry 
                    WHERE id > ? AND ai_risk_score IS NOT NULL
                    ORDER BY id ASC 
                    LIMIT 5
                """, (last_id,))
                
                rows = cur.fetchall()
                conn.close()
                
                for row in rows:
                    risk = row['ai_risk_score']
                    
                    await websocket.send_json({
                        "type": "reading",
                        "data": {
                            "id": row['id'],
                            "timestamp": str(row['timestamp_sim']),
                            "risk_score": risk,
                            "status": row['ai_status'] or "PENDING",
                            "pressure": row['quench_pressure']
                        }
                    })
                    last_id = row['id']
                    
            except Exception:
                pass 
            
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# =============================================================================
# SERVE FRONTEND (ANALYST ONLY)
# =============================================================================

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Serve the Analyst Dashboard."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Drift Sentinel</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f0f23 100%);
            color: #e0e0e0;
            padding: 20px;
            min-height: 100vh;
            overflow-y: auto; /* Main scrollbar enabled */
        }
        
        /* HEADER - Industrial Modern */
        .header { 
            display: flex; justify-content: space-between; align-items: center; 
            margin-bottom: 24px; padding-bottom: 16px;
            border-bottom: 2px solid rgba(100, 200, 255, 0.3);
        }
        .header h1 { 
            font-size: 1.6rem; font-weight: 600; color: #fff;
            display: flex; align-items: center; gap: 10px;
        }
        .header h1 .icon { font-size: 1.4rem; }
        .header h1 span { color: #4fc3f7; } /* Accent Blue */
        .live-indicator { 
            display: flex; align-items: center; gap: 8px; 
            font-size: 0.85rem; color: #fff; background: linear-gradient(90deg, #00c853, #69f0ae);
            padding: 8px 16px; border-radius: 20px; font-weight: 500;
        }
        .live-indicator::before { 
            content: ''; width: 8px; height: 8px; background: #fff; 
            border-radius: 50%; animation: pulse 1.5s infinite;
        }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
        
        .grid-container {
            display: grid;
            grid-template-columns: 300px 1fr 360px;
            gap: 20px;
            max-width: 1600px;
            margin: 0 auto;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 20px;
            display: flex; flex-direction: column;
        }
        .card h2 { 
            font-size: 0.8rem; color: #90a4ae; text-transform: uppercase; 
            margin-bottom: 16px; letter-spacing: 1px; font-weight: 600;
            display: flex; align-items: center; gap: 8px;
        }
        .card h2::before { content: '▸'; color: #4fc3f7; }
        
        /* SHIFT REPORT CARD */
        .score-box { text-align: center; padding: 20px 0; }
        .score-val { 
            font-size: 4.5rem; font-weight: 700; line-height: 1; 
        }
        .score-circle { 
            width: 140px; height: 140px; border-radius: 50%; 
            border: 4px solid rgba(255,255,255,0.1);
            display: flex; align-items: center; justify-content: center; margin: 0 auto;
            background: rgba(0,0,0,0.3);
        }
        .grade-info { 
            font-size: 0.75rem; color: #666; margin-top: 8px; 
            padding: 8px; background: rgba(0,0,0,0.3); border-radius: 6px;
        }
        .stat-row { 
            display: flex; justify-content: space-between; margin-top: 8px; 
            font-size: 0.85rem; border-bottom: 1px solid rgba(255,255,255,0.05); 
            padding: 8px 0;
        }
        .stat-row span:first-child { color: #888; }
        .stat-row span:last-child { font-weight: 600; font-family: 'Inter', sans-serif; }
        
        /* RCA DISPLAY - Modern Terminal */
        .rca-container { flex: 1; display: flex; flex-direction: column; gap: 16px; }
        .rca-display {
            background: rgba(0, 0, 0, 0.4); 
            border: 1px solid rgba(100, 200, 255, 0.2);
            border-radius: 8px;
            padding: 20px;
            font-family: 'Consolas', 'Monaco', monospace; 
            font-size: 0.85rem; line-height: 1.7;
            color: #b0bec5;
            flex: 1; overflow-y: auto;
            max-height: 300px;
        }
        .rca-display::before { 
            content: "⚡ DIAGNOSTICS"; display: block; color: #4fc3f7; 
            margin-bottom: 12px; font-weight: bold; font-size: 0.75rem;
            letter-spacing: 1px;
        }
        
        .chart-container { 
            height: 220px; background: rgba(0,0,0,0.3); 
            border: 1px solid rgba(255,255,255,0.08); border-radius: 8px;
            position: relative;
        }
        
        /* AUDIT LOG with Scrollbar */
        .audit-list { 
            overflow-y: auto; flex: 1; padding-right: 8px; 
            max-height: 400px;
        }
        .audit-item {
            padding: 12px; border-radius: 8px; cursor: pointer;
            transition: all 0.2s ease;
            margin-bottom: 8px; 
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.05);
        }
        .audit-item:hover { 
            background: rgba(79, 195, 247, 0.1); 
            border-color: rgba(79, 195, 247, 0.3);
        }
        .audit-header { display: flex; justify-content: space-between; margin-bottom: 6px; align-items: center; }
        .tag { padding: 3px 8px; font-size: 0.7rem; font-weight: 600; border-radius: 4px; }
        .tag-risk { background: rgba(255, 82, 82, 0.2); color: #ff5252; }
        .tag-time { color: #666; font-size: 0.75rem; }
        
        /* BUTTONS - Modern */
        .btn { 
            background: linear-gradient(135deg, #4fc3f7, #29b6f6);
            color: #000; border: none; padding: 10px 20px; 
            cursor: pointer; font-weight: 600; font-family: 'Inter', sans-serif;
            border-radius: 8px; font-size: 0.8rem;
            margin-top: 12px; align-self: flex-end;
            transition: all 0.2s ease;
        }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(79, 195, 247, 0.4); }
        
        /* Chart Legend */
        .chart-legend { 
            display: flex; gap: 20px; padding-top: 12px; font-size: 0.75rem; 
            color: #666;
        }
        .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
        .dot.green { background: #69f0ae; }
        .dot.yellow { background: #ffd54f; }
        .dot.red { background: #ff5252; }
        
        /* Modern Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); border-radius: 3px; }
        ::-webkit-scrollbar-thumb { background: rgba(79, 195, 247, 0.3); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(79, 195, 247, 0.5); }
        
        /* RESPONSIVE DESIGN */
        @media (max-width: 1200px) {
            .grid-container {
                grid-template-columns: 280px 1fr 300px;
                gap: 16px;
            }
        }
        
        @media (max-width: 1024px) {
            .grid-container {
                grid-template-columns: 1fr 1fr;
                grid-template-rows: auto auto auto;
            }
            .card:first-child { grid-column: 1; }
            .rca-container { grid-column: 2; grid-row: 1 / 3; }
            .card:last-child { grid-column: 1; grid-row: 2; }
        }
        
        @media (max-width: 768px) {
            body { padding: 12px; }
            .header { flex-direction: column; gap: 12px; text-align: center; }
            .header h1 { font-size: 1.3rem; }
            .grid-container {
                grid-template-columns: 1fr;
                grid-template-rows: auto;
            }
            .rca-container { grid-column: 1; grid-row: auto; }
            .card:last-child { grid-column: 1; grid-row: auto; }
            .score-val { font-size: 3.5rem; }
            .score-circle { width: 110px; height: 110px; }
            .chart-legend { flex-wrap: wrap; gap: 10px; }
        }
        
        @media (max-width: 480px) {
            .header h1 { font-size: 1.1rem; }
            .card { padding: 14px; }
            .card h2 { font-size: 0.7rem; }
            .grade-info { font-size: 0.65rem; }
            .stat-row { font-size: 0.8rem; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1><span class="icon">📡</span> <span>AI</span> Drift Sentinel</h1>
        <div class="live-indicator">LIVE</div>
    </div>
    
    <div class="grid-container">
        <!-- LEFT COL: SHIFT REPORT -->
        <div class="card">
            <h2>Shift Performance</h2>
            <div class="score-box">
                <div class="score-circle" id="scoreCircle">
                    <div class="score-val" id="shiftScore">--</div>
                </div>
                <p style="margin-top: 12px; color: #888; font-size: 0.9rem;" id="shiftSummary">Analyzing...</p>
            </div>
            <!-- Grade Range Info (Matches code logic line 182) -->
            <div class="grade-info">
                <strong style="color:#90a4ae;">Grade Scale (% Optimal):</strong><br>
                <span style="color:#69f0ae;">A</span> = &gt;95% &nbsp;|&nbsp;
                <span style="color:#ffd54f;">B</span> = 80-95% &nbsp;|&nbsp;
                <span style="color:#ff5252;">C</span> = &lt;80%
            </div>
            <div style="margin-top: auto; padding-top: 16px;">
                <div class="stat-row">
                    <span>Total Parts</span>
                    <span id="totalParts">0</span>
                </div>
                <div class="stat-row">
                    <span style="color:#69f0ae">● Optimal</span>
                    <span id="optPct">0%</span>
                </div>
                <div class="stat-row">
                    <span style="color:#ffd54f">● Warnings</span>
                    <span id="warnPct">0%</span>
                </div>
                <div class="stat-row">
                    <span style="color:#ff5252">● Critical</span>
                    <span id="critPct">0%</span>
                </div>
            </div>
        </div>
        
        <!-- CENTER COL: RCA & TRENDS -->
        <div class="rca-container">
            <!-- RCA Display -->
            <div class="card" style="flex: 1;">
                <h2>Automated RCA (Root Cause Analysis)</h2>
                <div class="rca-display" id="rcaText">
                    Waiting for event selection...
                </div>
                <button class="btn" onclick="analyzeLatest()">Analyze Latest Alert</button>
            </div>
            
            <!-- Drift Trend (Line Chart) -->
            <div class="card" style="height: 300px;">
                <h2>Drift Trend (Live)</h2>
                <div class="chart-container" style="position:relative;">
                    <!-- Chart renders here via JS -->
                    <div id="driftChart" style="width:100%;height:200px;position:relative;"></div>
                </div>
                <div class="chart-legend">
                    <span><span class="dot green"></span> Optimal (&lt;40%)</span>
                    <span><span class="dot yellow"></span> Warning (40-80%)</span>
                    <span><span class="dot red"></span> Critical (&gt;80%)</span>
                    <span style="margin-left:auto; color:#666;">← Older | Newer →</span>
                </div>
            </div>
        </div>
        
        <!-- RIGHT COL: AUDIT LOG -->
        <div class="card">
            <h2>Safety Alerts</h2>
            <p style="font-size: 0.8rem; color: #666; margin-bottom: 12px;">
                High-risk events detected by AI (Risk &gt; 80%)
            </p>
            <div class="audit-list" id="auditList">
                <!-- Latest alerts appear first -->
            </div>
        </div>
    </div>

    <script>
        let lastAlertId = 0;
        const trendHistory = [];
        
        // 1. Fetch Shift Report
        async function updateShiftReport() {
            try {
                const res = await fetch('/api/reports/shift');
                const data = await res.json();
                
                const scoreEl = document.getElementById('shiftScore');
                const circEl = document.getElementById('scoreCircle');
                
                scoreEl.textContent = data.shift_score;
                scoreEl.style.color = data.score_color;
                circEl.style.borderColor = data.score_color;
                
                document.getElementById('shiftSummary').innerHTML = data.summary;
                document.getElementById('totalParts').textContent = data.total_parts;
                document.getElementById('optPct').textContent = data.health_stats.optimal;
                document.getElementById('warnPct').textContent = data.health_stats.warning;
                document.getElementById('critPct').textContent = data.health_stats.critical;
            } catch(e) { console.log(e); }
        }
        
        // 2. Automated RCA
        async function fetchRCA(id) {
            document.getElementById('rcaText').innerHTML = "⏳ Mining DB history for Event #" + id + "...";
            try {
                const res = await fetch(`/api/rca/${id}`);
                const data = await res.json();
                document.getElementById('rcaText').innerHTML = data.explanation || "No analysis available.";
            } catch(e) {
                document.getElementById('rcaText').innerHTML = "❌ Analysis Failed: " + e;
            }
        }
        
        // 3. Safety Audit Log (Latest First)
        async function updateAuditLog() {
            try {
                const res = await fetch('/api/audit/safety_net');
                const data = await res.json();
                const list = document.getElementById('auditList');
                list.innerHTML = '';
                
                // Sort by ID descending (latest first) 
                const sortedLog = data.audit_log.sort((a, b) => b.id - a.id);
                
                sortedLog.forEach(item => {
                    const el = document.createElement('div');
                    el.className = 'audit-item';
                    el.onclick = () => fetchRCA(item.id);
                    el.innerHTML = `
                        <div class="audit-header">
                            <span class="tag tag-risk">⚠️ ${(item.risk*100).toFixed(0)}%</span>
                            <span class="tag-time">ID #${item.id}</span>
                        </div>
                        <div style="font-size: 0.8rem; color: #b0bec5; line-height: 1.4;">${item.msg}</div>
                    `;
                    list.appendChild(el);
                    
                    // Track latest for auto-analyze
                    if (item.id > lastAlertId) lastAlertId = item.id;
                });
            } catch(e) { console.log(e); }
        }
        
        async function analyzeLatest() {
            // Get latest high-risk event from Audit Log
            const res = await fetch('/api/audit/safety_net');
            const data = await res.json();
            if(data.audit_log && data.audit_log.length > 0) {
                fetchRCA(data.audit_log[0].id);
            } else {
                 document.getElementById('rcaText').innerHTML = "No critical events found to analyze.";
            }
        }

        // 4. WebSocket for Drift Trend
        function connectWs() {
            const ws = new WebSocket(`ws://${location.host}/ws`);
            ws.onmessage = (e) => {
                const msg = JSON.parse(e.data);
                if(msg.type === 'reading') {
                    const d = msg.data;
                    trendHistory.push(d.risk_score);
                    if(trendHistory.length > 100) trendHistory.shift();
                    renderChart();
                }
            };
            ws.onclose = () => setTimeout(connectWs, 3000);
        }
        
        function renderChart() {
            const container = document.getElementById('driftChart');
            const width = container.clientWidth || 600;
            const height = container.clientHeight || 180;
            
            if (trendHistory.length < 2) {
                container.innerHTML = '<div style="color:#444;text-align:center;padding-top:80px;">Waiting for data...</div>';
                return;
            }
            
            // Get color based on value
            const getColor = (val) => val > 0.8 ? '#ff5252' : (val > 0.4 ? '#ffd54f' : '#69f0ae');
            
            // Build line segments with individual colors
            let lineSegments = '';
            for (let i = 0; i < trendHistory.length - 1; i++) {
                const x1 = (i / (trendHistory.length - 1)) * width;
                const y1 = height - (trendHistory[i] * height);
                const x2 = ((i + 1) / (trendHistory.length - 1)) * width;
                const y2 = height - (trendHistory[i + 1] * height);
                
                // Use the max value of the two points to determine segment color
                const maxVal = Math.max(trendHistory[i], trendHistory[i + 1]);
                const color = getColor(maxVal);
                
                lineSegments += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" 
                                       stroke="${color}" stroke-width="2" stroke-linecap="round"/>`;
            }
            
            // Current value for overlay
            const currentVal = trendHistory[trendHistory.length - 1];
            const currentPct = (currentVal * 100).toFixed(0);
            const currentColor = getColor(currentVal);
            
            // Create SVG
            container.innerHTML = `
                <svg width="100%" height="100%" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
                    <!-- Threshold lines -->
                    <line x1="0" y1="${height * 0.2}" x2="${width}" y2="${height * 0.2}" 
                          stroke="#ff5252" stroke-width="1" stroke-dasharray="4,4" opacity="0.6"/>
                    <line x1="0" y1="${height * 0.6}" x2="${width}" y2="${height * 0.6}" 
                          stroke="#ffd54f" stroke-width="1" stroke-dasharray="4,4" opacity="0.6"/>
                    
                    <!-- Color-coded line segments -->
                    ${lineSegments}
                    
                    <!-- Current value dot (pulsing) -->
                    <circle cx="${width}" cy="${height - (currentVal * height)}" r="5" 
                            fill="${currentColor}" stroke="#fff" stroke-width="1.5">
                        <animate attributeName="r" values="5;7;5" dur="1s" repeatCount="indefinite"/>
                    </circle>
                </svg>
                
                <!-- Current value overlay -->
                <div style="position:absolute;top:10px;right:10px;background:rgba(0,0,0,0.85);
                            padding:8px 14px;border-radius:6px;border:2px solid ${currentColor};
                            font-size:1.3rem;color:${currentColor};font-weight:700;">
                    ${currentPct}%
                </div>
            `;
        }
        
        // Init
        updateShiftReport();
        updateAuditLog();
        connectWs();
        setInterval(() => {
            updateShiftReport();
            updateAuditLog();
        }, 5000);
        
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🧠 AI ANALYST DASHBOARD STARTED")
    print("=" * 60)
    print(f"   URL:  http://localhost:{DASHBOARD_PORT}")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=DASHBOARD_PORT)
