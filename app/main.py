"""
ArduPilot AI Tuning System - Main Application
A closed-loop system that analyzes flight logs and recommends optimal parameters.
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum

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
VEHICLES_FILE = DATA_DIR / "vehicles.json"
LOGS_DIR = DATA_DIR / "logs"
OUTPUTS_DIR = DATA_DIR / "outputs"
TRAINING_DIR = DATA_DIR / "training_jobs"

# Ensure directories exist
LOGS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
TRAINING_DIR.mkdir(exist_ok=True)


class DroneType(str, Enum):
    FPV_RACING = "fpv_racing"
    CINEMATIC = "cinematic"
    SURVEY = "survey"
    AGRICULTURAL = "agricultural"
    HEAVY_LIFT = "heavy_lift"
    VTOL = "vtol"
    FIXED_WING = "fixed_wing"
    ROVER = "rover"
    HELICOPTER = "helicopter"


# Vehicle type descriptions and focus areas
DRONE_TYPE_INFO = {
    "fpv_racing": {
        "name": "FPV Racing Drone",
        "description": "Fast, agile drone for racing and freestyle",
        "focus_params": ["ATC_RAT_RLL_P", "ATC_RAT_RLL_D", "ATC_RAT_YAW_P", "ATC_RAT_RLL_MAX"],
        "typical_issues": ["Overshoot", "High vibration", "Yaw wobble"]
    },
    "cinematic": {
        "name": "Cinematic/Aerial Photography",
        "description": "Camera drone requiring smooth, stable flight",
        "focus_params": ["ATC_RAT_PIT_P", "ATC_ANG_PIT_P", "PSC_POS_Z_P", "PSC_VEL_Z_P"],
        "typical_issues": ["Jitter", "Altitude drift", "Oscillations"]
    },
    "survey": {
        "name": "Survey/Mapping Drone",
        "description": "Precision mapping with GPS accuracy",
        "focus_params": ["EK3_GPS_GAIN", "PSC_POS_X_P", "FENCE_ENABLE", "WPNAV_SPEED"],
        "typical_issues": ["GPS drift", "Position error", "Waypoint overshoot"]
    },
    "agricultural": {
        "name": "Agricultural Sprayer",
        "description": "Heavy payload spraying drone",
        "focus_params": ["MOT_BAT_VOLT_MAX", "MOT_SPIN_MIN", "ATC_RAT_RLL_P", "INS_GYRO_FILTER"],
        "typical_issues": ["Motor overheating", "Vibration", "Payload handling"]
    },
    "heavy_lift": {
        "name": "Heavy Lift Drone",
        "description": "Large drone carrying heavy payloads",
        "focus_params": ["ATC_RAT_RLL_P", "ATC_RAT_RLL_I", "INS_GYRO_FILTER", "PSC_ACCZ_P"],
        "typical_issues": ["Overshoot", "Slow response", "Vibration"]
    },
    "vtol": {
        "name": "VTOL Aircraft",
        "description": "Vertical takeoff and landing aircraft",
        "focus_params": ["VT_FW_Q_SPEED", "VT_TRANS_TIME", "TECS_PITCH_MAX", "CRUISE_SPEED"],
        "typical_issues": ["Transition issues", "Forward flight handling"]
    },
    "fixed_wing": {
        "name": "Fixed Wing Plane",
        "description": "Traditional airplane autopilot",
        "focus_params": ["TECS_SPDWEIGHT", "CRUISE_SPEED", "RTL_SPEED", "WPNAV_SPEED"],
        "typical_issues": ["Speed control", "Altitude overshoot", "Landing approach"]
    },
    "rover": {
        "name": "Rover/Ground Vehicle",
        "description": "Ground-based autonomous vehicle",
        "focus_params": ["CRUISE_SPEED", "SERVO5_FUNCTION", "RC1_TRIM", "RC2_TRIM"],
        "typical_issues": ["Wheel slip", "Speed overshoot"]
    },
    "helicopter": {
        "name": "Helicopter",
        "description": "Single rotor helicopter",
        "focus_params": ["HELI_GOV_ENABLE", "ATC_RAT_RLL_P", "ATC_RAT_PIT_P", "RCOUT_MIN"],
        "typical_issues": ["Rotor RPM", "Governor tuning", "Collective response"]
    }
}

# In-memory storage for active vehicle (in production, use database)
current_vehicle = None

# Training job status
training_jobs = {}


def load_vehicles() -> List[Dict]:
    """Load vehicles from file."""
    if VEHICLES_FILE.exists():
        with open(VEHICLES_FILE) as f:
            return json.load(f)
    return []


def save_vehicles(vehicles: List[Dict]):
    """Save vehicles to file."""
    with open(VEHICLES_FILE, "w") as f:
        json.dump(vehicles, f, indent=2)


app = FastAPI(
    title="ArduPilot AI Tuner",
    description="AI-powered parameter tuning system for ArduPilot drones",
    version="1.0.0"
)

# Setup templates
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main interface - setup vehicle first, then upload logs."""
    global current_vehicle
    vehicles = load_vehicles()
    
    # Check for active vehicle
    active_vehicle = None
    if current_vehicle:
        active_vehicle = current_vehicle
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "vehicles": vehicles,
        "active_vehicle": active_vehicle,
        "drone_types": DRONE_TYPE_INFO
    })


@app.post("/vehicle/create")
async def create_vehicle(
    name: str = Form(...),
    drone_type: str = Form(...),
    description: Optional[str] = Form(None),
    current_params: Optional[str] = Form(None)
):
    """Create a new vehicle profile."""
    global current_vehicle
    
    vehicles = load_vehicles()
    
    # Get drone type info
    type_info = DRONE_TYPE_INFO.get(drone_type, DRONE_TYPE_INFO["fpv_racing"])
    
    new_vehicle = {
        "id": len(vehicles) + 1,
        "name": name,
        "drone_type": drone_type,
        "drone_type_name": type_info["name"],
        "description": description or "",
        "current_params": current_params or "",
        "focus_params": type_info["focus_params"],
        "created_at": datetime.now().isoformat(),
        "flight_count": 0
    }
    
    vehicles.append(new_vehicle)
    save_vehicles(vehicles)
    
    # Set as active vehicle
    current_vehicle = new_vehicle
    
    return JSONResponse({"status": "success", "vehicle": new_vehicle})


@app.post("/vehicle/select")
async def select_vehicle(vehicle_id: int = Form(...)):
    """Select an existing vehicle as active."""
    global current_vehicle
    
    vehicles = load_vehicles()
    for vehicle in vehicles:
        if vehicle["id"] == vehicle_id:
            current_vehicle = vehicle
            return JSONResponse({"status": "success", "vehicle": vehicle})
    
    raise HTTPException(status_code=404, detail="Vehicle not found")


@app.get("/vehicle/current")
async def get_current_vehicle():
    """Get the currently active vehicle."""
    global current_vehicle
    if current_vehicle:
        return JSONResponse(current_vehicle)
    return JSONResponse({"status": "no_vehicle"})


@app.post("/vehicle/delete")
async def delete_vehicle(vehicle_id: int = Form(...)):
    """Delete a vehicle profile."""
    global current_vehicle
    
    vehicles = load_vehicles()
    vehicles = [v for v in vehicles if v["id"] != vehicle_id]
    save_vehicles(vehicles)
    
    # Clear current if deleted
    if current_vehicle and current_vehicle["id"] == vehicle_id:
        current_vehicle = None
    
    return JSONResponse({"status": "success"})


@app.post("/analyze")
async def analyze_log(
    file: UploadFile = File(...),
    notes: Optional[str] = Form(None)
):
    """Analyze a flight log and return parameter recommendations."""
    global current_vehicle
    
    # Check if vehicle is selected
    if not current_vehicle:
        raise HTTPException(
            status_code=400,
            detail="Please create or select a vehicle first before uploading logs."
        )
    
    # Validate file extension
    if not file.filename.endswith(('.bin', '.log')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload .bin or .log files."
        )
    
    # Save uploaded file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vehicle_prefix = current_vehicle["drone_type"]
    filepath = LOGS_DIR / f"{vehicle_prefix}_{timestamp}_{file.filename}"
    
    try:
        with open(filepath, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Analyze the log
        metrics = analyze_flight_log(str(filepath))
        
        # Generate recommendations based on vehicle type
        recommendations = generate_recommendations(
            metrics, 
            current_vehicle["drone_type"],
            notes
        )
        
        # Save output
        output_file = OUTPUTS_DIR / f"{timestamp}_analysis.json"
        output_data = {
            "vehicle_id": current_vehicle["id"],
            "vehicle_name": current_vehicle["name"],
            "drone_type": current_vehicle["drone_type"],
            "metrics": metrics,
            "recommendations": recommendations,
            "notes": notes,
            "timestamp": timestamp
        }
        
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        
        # Update vehicle flight count
        vehicles = load_vehicles()
        for v in vehicles:
            if v["id"] == current_vehicle["id"]:
                v["flight_count"] = v.get("flight_count", 0) + 1
        save_vehicles(vehicles)
        
        return JSONResponse({
            "status": "success",
            "vehicle": current_vehicle["name"],
            "drone_type": current_vehicle["drone_type_name"],
            "metrics": metrics,
            "recommendations": recommendations,
            "log_file": str(filepath)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def generate_recommendations(metrics: Dict, drone_type: str, notes: Optional[str] = None) -> List[Dict]:
    """Generate parameter recommendations based on flight metrics and drone type."""
    recommendations = []
    
    # Get focus parameters for this drone type
    type_info = DRONE_TYPE_INFO.get(drone_type, DRONE_TYPE_INFO["fpv_racing"])
    focus_params = type_info["focus_params"]
    
    # Extract key metrics with defaults
    roll_error = metrics.get("max_roll_error", 0)
    pitch_error = metrics.get("max_pitch_error", 0)
    overshoot = metrics.get("overshoot_percent", 0)
    settling_time = metrics.get("settling_time_sec", 0)
    vibration = metrics.get("vibration_g", 0)
    roll_rate = metrics.get("max_roll_rate", 0)
    
    # Drone-type specific tuning
    if drone_type == "fpv_racing":
        if overshoot > 8:
            recommendations.append({
                "parameter": "ATC_RAT_RLL_P",
                "current": "0.12",
                "recommended": "0.08",
                "change": "-33%",
                "reason": "FPV racing needs less overshoot for precise control"
            })
        if vibration > 2.0:
            recommendations.append({
                "parameter": "INS_HNTC2_ENABLE",
                "current": "0",
                "recommended": "1",
                "change": "Enable",
                "reason": "Enable harmonic notch filter for high-speed flight"
            })
    
    elif drone_type == "cinematic":
        if pitch_error > 10:
            recommendations.append({
                "parameter": "ATC_RAT_PIT_P",
                "current": "0.12",
                "recommended": "0.06",
                "change": "-50%",
                "reason": "Cinematic needs smooth, gentle response"
            })
        if metrics.get("vibration_g", 0) > 1.5:
            recommendations.append({
                "parameter": "INS_GYRO_FILTER",
                "current": "80",
                "recommended": "60",
                "change": "-25%",
                "reason": "Reduce vibrations for stable video"
            })
        if settling_time > 1.5:
            recommendations.append({
                "parameter": "PSC_VEL_Z_P",
                "current": "5.0",
                "recommended": "2.5",
                "change": "-50%",
                "reason": "Gentler altitude control for smooth footage"
            })
    
    elif drone_type == "survey":
        if metrics.get("gps_horizontal_accuracy", 0) > 1.0:
            recommendations.append({
                "parameter": "EK3_GPS_GAIN",
                "current": "1.0",
                "recommended": "0.6",
                "change": "-40%",
                "reason": "Reduce GPS weight for better accuracy"
            })
        if roll_error > 5:
            recommendations.append({
                "parameter": "WPNAV_RADIUS",
                "current": "2",
                "recommended": "1",
                "change": "-50%",
                "reason": "Tighter waypoints for survey accuracy"
            })
    
    elif drone_type == "agricultural":
        if metrics.get("max_roll_error", 0) > 15:
            recommendations.append({
                "parameter": "ATC_RAT_RLL_P",
                "current": "0.12",
                "recommended": "0.07",
                "change": "-42%",
                "reason": "Heavy payload needs gentler response"
            })
        if vibration > 3.0:
            recommendations.append({
                "parameter": "INS_GYRO_FILTER",
                "current": "80",
                "recommended": "40",
                "change": "-50%",
                "reason": "Critical vibration reduction for sprayer"
            })
    
    elif drone_type == "vtol":
        if overshoot > 10:
            recommendations.append({
                "parameter": "VT_FW_Q_SPEED",
                "current": "15",
                "recommended": "22",
                "change": "+47%",
                "reason": "Higher transition speed for cleaner switch"
            })
    
    # General PID rules (apply to all if not covered above)
    if overshoot > 10:
        # Only add if not already added for specific type
        param_exists = any(r["parameter"] == "ATC_RAT_RLL_P" for r in recommendations)
        if not param_exists:
            recommendations.append({
                "parameter": "ATC_RAT_RLL_P",
                "current": "0.12",
                "recommended": "0.09",
                "change": "-25%",
                "reason": "Reduce P gain to decrease overshoot"
            })
    
    if settling_time > 2.0:
        param_exists = any(r["parameter"] == "ATC_RAT_RLL_I" for r in recommendations)
        if not param_exists:
            recommendations.append({
                "parameter": "ATC_RAT_RLL_I",
                "current": "0.05",
                "recommended": "0.08",
                "change": "+60%",
                "reason": "Increase I term for faster settling"
            })
    
    if vibration > 2.5:
        param_exists = any(r["parameter"] == "INS_GYRO_FILTER" for r in recommendations)
        if not param_exists:
            recommendations.append({
                "parameter": "INS_GYRO_FILTER",
                "current": "80",
                "recommended": "40",
                "change": "-50%",
                "reason": "Lower filter cutoff to reduce vibration impact"
            })
    
    return recommendations


@app.get("/history")
async def get_history(vehicle_id: int = None):
    """Get history of all analyzed logs for a vehicle."""
    history = []
    
    for output_file in OUTPUTS_DIR.glob("*_analysis.json"):
        with open(output_file) as f:
            data = json.load(f)
            if vehicle_id is None or data.get("vehicle_id") == vehicle_id:
                history.append({
                    "timestamp": data.get("timestamp"),
                    "vehicle_name": data.get("vehicle_name"),
                    "drone_type": data.get("drone_type"),
                    "metrics": data.get("metrics", {}),
                    "recommendations": data.get("recommendations", [])
                })
    
    # Sort by timestamp descending
    history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    return JSONResponse(history)


@app.get("/vehicles")
async def list_vehicles():
    """List all vehicles."""
    vehicles = load_vehicles()
    return JSONResponse(vehicles)


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


# Training endpoints
@app.post("/training/start")
async def start_training(
    model_name: str = Form(...),
    epochs: int = Form(3)
):
    """Start fine-tuning the model."""
    global training_jobs
    
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    training_jobs[job_id] = {
        "status": "starting",
        "model": model_name,
        "epochs": epochs,
        "started_at": datetime.now().isoformat(),
        "progress": 0
    }
    
    # In production, this would start actual training in background
    return JSONResponse({
        "status": "started",
        "job_id": job_id,
        "message": f"Training started with {model_name} for {epochs} epochs"
    })


@app.get("/training/status")
async def get_training_status(job_id: str = None):
    """Get training status."""
    if job_id:
        if job_id in training_jobs:
            return JSONResponse(training_jobs[job_id])
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(list(training_jobs.values()))


@app.post("/training/stop")
async def stop_training(job_id: str = Form(...)):
    """Stop a training job."""
    if job_id in training_jobs:
        training_jobs[job_id]["status"] = "stopped"
        return JSONResponse({
            "status": "stopped",
            "job_id": job_id,
            "message": "Training job stopped"
        })
    raise HTTPException(status_code=404, detail="Job not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)