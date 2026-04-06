"""
ArduPilot AI Tuning System - Main Application
A closed-loop system that analyzes flight logs and recommends optimal parameters.
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

# Import local modules
from analyzer import analyze_flight_log, extract_key_metrics
from parameters import ARDUPILOT_PARAMETERS, validate_parameter, format_recommendations

# Configuration
DATA_DIR = Path("data")
LOGS_DIR = DATA_DIR / "logs"
OUTPUTS_DIR = DATA_DIR / "outputs"

# Ensure directories exist
LOGS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="ArduPilot AI Tuner",
    description="AI-powered parameter tuning system for ArduPilot drones",
    version="1.0.0"
)

# Setup templates
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main interface - upload flight logs and view recommendations."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze")
async def analyze_log(
    file: UploadFile = File(...),
    context: Optional[str] = Form(None)
):
    """
    Analyze a flight log and return parameter recommendations.
    
    Args:
        file: ArduPilot flight log (.bin or .log)
        context: Optional context about the drone (frame type, use case, etc.)
    
    Returns:
        JSON with analysis and parameter recommendations
    """
    # Validate file extension
    if not file.filename.endswith(('.bin', '.log')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload .bin or .log files."
        )
    
    # Save uploaded file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = LOGS_DIR / f"{timestamp}_{file.filename}"
    
    try:
        with open(filepath, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Analyze the log
        metrics = analyze_flight_log(str(filepath))
        
        # Generate recommendations (simulated - integrate with Ollama in production)
        recommendations = generate_recommendations(metrics, context)
        
        # Save output
        output_file = OUTPUTS_DIR / f"{timestamp}_analysis.json"
        with open(output_file, "w") as f:
            json.dump({
                "metrics": metrics,
                "recommendations": recommendations,
                "timestamp": timestamp
            }, f, indent=2)
        
        return JSONResponse({
            "status": "success",
            "metrics": metrics,
            "recommendations": recommendations,
            "log_file": str(filepath)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def generate_recommendations(metrics: Dict, context: Optional[str]) -> List[Dict]:
    """
    Generate parameter recommendations based on flight metrics.
    
    In production, this would call the fine-tuned Ollama model.
    For now, returns rule-based recommendations.
    """
    recommendations = []
    
    # Extract key metrics with defaults
    roll_error = metrics.get("max_roll_error", 0)
    pitch_error = metrics.get("max_pitch_error", 0)
    overshoot = metrics.get("overshoot_percent", 0)
    settling_time = metrics.get("settling_time_sec", 0)
    vibration = metrics.get("vibration_g", 0)
    
    # Roll PID tuning
    if overshoot > 10:
        recommendations.append({
            "parameter": "ATC_RAT_RLL_P",
            "current": "0.12",
            "recommended": "0.09",
            "change": "-25%",
            "reason": "High overshoot indicates P gain too high"
        })
    
    if settling_time > 2.5:
        recommendations.append({
            "parameter": "ATC_RAT_RLL_I",
            "current": "0.05",
            "recommended": "0.08",
            "change": "+60%",
            "reason": "Increase I term to reduce steady-state error faster"
        })
    
    # D term for damping
    if overshoot > 5:
        recommendations.append({
            "parameter": "ATC_RAT_RLL_D",
            "current": "0.002",
            "recommended": "0.004",
            "change": "+100%",
            "reason": "Add damping to reduce oscillation"
        })
    
    # Vibration handling
    if vibration > 2.5:
        recommendations.append({
            "parameter": "INS_GYRO_FILTER",
            "current": "80",
            "recommended": "40",
            "change": "-50%",
            "reason": "Reduce filter cutoff to reduce vibration impact"
        })
    
    # Rate limits
    if metrics.get("max_roll_rate", 0) > 500:
        recommendations.append({
            "parameter": "ATC_RAT_RLL_MAX",
            "current": "7200",
            "recommended": "5400",
            "change": "-25%",
            "reason": "Reduce max rate to prevent aggressive maneuvers"
        })
    
    return recommendations


@app.get("/history")
async def get_history():
    """Get history of all analyzed logs and recommendations."""
    history = []
    
    for output_file in OUTPUTS_DIR.glob("*_analysis.json"):
        with open(output_file) as f:
            data = json.load(f)
            history.append({
                "timestamp": data.get("timestamp"),
                "metrics": data.get("metrics", {}),
                "recommendations": data.get("recommendations", [])
            })
    
    # Sort by timestamp descending
    history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    return JSONResponse(history)


@app.get("/parameters")
async def list_parameters():
    """List all tuneable ArduPilot parameters with their ranges."""
    return JSONResponse(ARDUPILOT_PARAMETERS)


@app.post("/parameters/validate")
async def validate_param(
    parameter: str = Form(...),
    value: str = Form(...)
):
    """Validate if a parameter value is within safe bounds."""
    is_valid, message = validate_parameter(parameter, value)
    
    if is_valid:
        return JSONResponse({"status": "valid", "message": message})
    else:
        raise HTTPException(status_code=400, detail=message)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)