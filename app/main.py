"""
ArduPilot AI Tuning System - Main Application
A closed-loop system that analyzes flight logs and recommends optimal parameters.
"""

import json
import re
import uuid
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum
import asyncio
from asyncio.subprocess import Process
from contextlib import suppress
from dataclasses import dataclass

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import httpx

# Import local modules
from analyzer import analyze_flight_log
from parameters import ARDUPILOT_PARAMETERS, validate_parameter

# Configuration
DATA_DIR = Path("data")
VEHICLES_FILE = DATA_DIR / "vehicles.json"
LOGS_DIR = DATA_DIR / "logs"
OUTPUTS_DIR = DATA_DIR / "outputs"
TRAINING_DIR = DATA_DIR / "training_jobs"
ACTIVE_VEHICLE_COOKIE = "active_vehicle_id"
ALLOWED_LOG_EXTENSIONS = {".bin", ".log"}
MAX_UPLOAD_SIZE_BYTES = 200 * 1024 * 1024
UPLOAD_CHUNK_SIZE = 1024 * 1024

MODEL_ALIASES = {
    "qwen2.5:7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5:14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5:72b": "Qwen/Qwen2.5-72B-Instruct",
}

# Ensure directories exist
LOGS_DIR.mkdir(exist_ok=True, parents=True)
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
TRAINING_DIR.mkdir(exist_ok=True, parents=True)


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

# In-memory training state
training_jobs: Dict[str, Dict] = {}
training_processes: Dict[str, Process] = {}
training_tasks: Dict[str, asyncio.Task] = {}


def _build_parameter_index() -> Dict[str, Dict]:
    index: Dict[str, Dict] = {}
    for category in ARDUPILOT_PARAMETERS.values():
        for param_name, param_info in category.get("parameters", {}).items():
            index[param_name] = param_info
    return index


PARAMETER_INDEX = _build_parameter_index()


@dataclass
class ParameterSuggestion:
    parameter: str
    target_value: float
    reason: str


def load_vehicles() -> List[Dict]:
    """Load vehicles from file."""
    if VEHICLES_FILE.exists():
        try:
            with open(VEHICLES_FILE, encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Failed to decode {VEHICLES_FILE}. Returning empty list.")
            return []
    return []


def save_vehicles(vehicles: List[Dict]):
    """Save vehicles to file."""
    with open(VEHICLES_FILE, "w", encoding="utf-8") as f:
        json.dump(vehicles, f, indent=2)


def get_vehicle_by_id(vehicle_id: int) -> Optional[Dict]:
    for vehicle in load_vehicles():
        if vehicle.get("id") == vehicle_id:
            return vehicle
    return None


def get_active_vehicle_from_request(request: Request) -> Optional[Dict]:
    raw_vehicle_id = request.cookies.get(ACTIVE_VEHICLE_COOKIE)
    if not raw_vehicle_id:
        return None

    try:
        vehicle_id = int(raw_vehicle_id)
    except (TypeError, ValueError):
        return None

    return get_vehicle_by_id(vehicle_id)


def sanitize_upload_filename(filename: str) -> str:
    base_name = Path(filename).name
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", base_name).lstrip(".")
    return safe_name or "uploaded_log.log"


async def save_uploaded_file(file: UploadFile, destination: Path) -> int:
    total_bytes = 0
    with open(destination, "wb") as f:
        while True:
            chunk = await file.read(UPLOAD_CHUNK_SIZE)
            if not chunk:
                break

            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_SIZE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum allowed size is {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB.",
                )
            f.write(chunk)
    return total_bytes


def parse_current_params(raw_params: Optional[str]) -> Dict[str, float]:
    parsed: Dict[str, float] = {}
    if not raw_params:
        return parsed

    for chunk in re.split(r"[,\n;]+", raw_params):
        token = chunk.strip()
        if not token or "=" not in token:
            continue
        key, raw_value = token.split("=", 1)
        key = key.strip().upper()
        raw_value = raw_value.strip()
        try:
            parsed[key] = float(raw_value)
        except ValueError:
            continue

    return parsed


def get_parameter_default(parameter: str) -> Optional[float]:
    info = PARAMETER_INDEX.get(parameter)
    if not info:
        return None
    try:
        return float(info.get("default"))
    except (TypeError, ValueError):
        return None


def clamp_parameter_value(parameter: str, value: float) -> float:
    info = PARAMETER_INDEX.get(parameter)
    if not info:
        return value
    low, high = info.get("range", [value, value])
    return max(float(low), min(float(high), value))


def format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def normalize_model_name(model_name: str) -> str:
    normalized = model_name.strip()
    return MODEL_ALIASES.get(normalized, normalized)


def build_recommendation(
    parameter: str,
    current_value: float,
    recommended_value: float,
    reason: str
) -> Dict:
    info = PARAMETER_INDEX.get(parameter, {})
    recommendation = {
        "parameter": parameter,
        "current": format_number(current_value),
        "recommended": format_number(recommended_value),
        "reason": reason,
    }

    if info.get("unit") == "bool":
        recommendation["change"] = "Enable" if recommended_value >= 1 else "Disable"
        return recommendation

    if current_value == 0:
        recommendation["change"] = "Set value"
        return recommendation

    delta_pct = ((recommended_value - current_value) / current_value) * 100.0
    recommendation["change"] = f"{delta_pct:+.0f}%"
    return recommendation


def add_recommendation(
    recommendations: List[Dict],
    current_params: Dict[str, float],
    suggestion: ParameterSuggestion
):
    parameter = suggestion.parameter

    if any(rec.get("parameter") == parameter for rec in recommendations):
        return

    current_value = current_params.get(parameter)
    if current_value is None:
        current_value = get_parameter_default(parameter)
    if current_value is None:
        return

    recommended_value = clamp_parameter_value(parameter, suggestion.target_value)
    if abs(recommended_value - current_value) < 1e-9:
        return

    recommendations.append(
        build_recommendation(
            parameter=parameter,
            current_value=current_value,
            recommended_value=recommended_value,
            reason=suggestion.reason,
        )
    )


def percent_adjustment(value: float, multiplier: float) -> float:
    return value * multiplier


app = FastAPI(
    title="ArduPilot AI Tuner",
    description="AI-powered parameter tuning system for ArduPilot drones",
    version="1.0.0"
)

# Setup templates
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main interface - setup vehicle first, then upload logs."""
    vehicles = load_vehicles()
    active_vehicle = get_active_vehicle_from_request(request)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "vehicles": vehicles,
            "active_vehicle": active_vehicle,
            "drone_types": DRONE_TYPE_INFO
        }
    )


@app.post("/vehicle/create")
async def create_vehicle(
    name: str = Form(...),
    drone_type: str = Form(...),
    description: Optional[str] = Form(None),
    current_params: Optional[str] = Form(None)
):
    """Create a new vehicle profile."""
    vehicles = load_vehicles()

    # Get drone type info
    type_info = DRONE_TYPE_INFO.get(drone_type, DRONE_TYPE_INFO["fpv_racing"])

    new_id = max((v["id"] for v in vehicles), default=0) + 1
    new_vehicle = {
        "id": new_id,
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

    response = JSONResponse({"status": "success", "vehicle": new_vehicle})
    response.set_cookie(
        key=ACTIVE_VEHICLE_COOKIE,
        value=str(new_id),
        httponly=True,
        samesite="lax",
    )
    return response


@app.post("/vehicle/select")
async def select_vehicle(vehicle_id: int = Form(...)):
    """Select an existing vehicle as active."""
    vehicles = load_vehicles()
    for vehicle in vehicles:
        if vehicle["id"] == vehicle_id:
            response = JSONResponse({"status": "success", "vehicle": vehicle})
            response.set_cookie(
                key=ACTIVE_VEHICLE_COOKIE,
                value=str(vehicle_id),
                httponly=True,
                samesite="lax",
            )
            return response

    raise HTTPException(status_code=404, detail="Vehicle not found")


@app.get("/vehicle/current")
async def get_current_vehicle(request: Request):
    """Get the currently active vehicle."""
    vehicle = get_active_vehicle_from_request(request)
    if vehicle:
        return JSONResponse(vehicle)
    return JSONResponse({"status": "no_vehicle"})


@app.post("/vehicle/delete")
async def delete_vehicle(request: Request, vehicle_id: int = Form(...)):
    """Delete a vehicle profile."""
    vehicles = load_vehicles()
    vehicles = [v for v in vehicles if v["id"] != vehicle_id]
    save_vehicles(vehicles)

    response = JSONResponse({"status": "success"})
    active_id = request.cookies.get(ACTIVE_VEHICLE_COOKIE)
    if active_id == str(vehicle_id):
        response.delete_cookie(ACTIVE_VEHICLE_COOKIE)
    return response


@app.post("/analyze")
async def analyze_log(
    request: Request,
    file: UploadFile = File(...),
    notes: Optional[str] = Form(None),
    model: str = Form("qwen2.5:7b")
):
    """Analyze a flight log and return parameter recommendations."""
    current_vehicle = get_active_vehicle_from_request(request)


    # Check if vehicle is selected
    if not current_vehicle:
        raise HTTPException(
            status_code=400,
            detail="Please create or select a vehicle first before uploading logs."
        )

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided.")

    safe_file_name = sanitize_upload_filename(file.filename)
    file_extension = Path(safe_file_name).suffix.lower()

    # Validate file extension
    if file_extension not in ALLOWED_LOG_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload .bin or .log files."
        )

    # Save uploaded file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    unique_suffix = uuid.uuid4().hex[:8]
    vehicle_prefix = current_vehicle["drone_type"]
    filepath = LOGS_DIR / f"{vehicle_prefix}_{timestamp}_{unique_suffix}_{safe_file_name}"

    try:
        written_bytes = await save_uploaded_file(file, filepath)
        if written_bytes <= 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Analyze the log
        metrics = await asyncio.to_thread(analyze_flight_log, str(filepath))

        # Generate recommendations based on vehicle type
        current_params = parse_current_params(current_vehicle.get("current_params"))
        recommendations = await generate_recommendations(
            metrics,
            current_vehicle["drone_type"],
            notes,
            current_params,
            model,
        )

        # Save output
        output_file = OUTPUTS_DIR / f"{timestamp}_{unique_suffix}_analysis.json"
        output_data = {
            "vehicle_id": current_vehicle["id"],
            "vehicle_name": current_vehicle["name"],
            "drone_type": current_vehicle["drone_type"],
            "metrics": metrics,
            "recommendations": recommendations,
            "notes": notes,
            "timestamp": timestamp,
            "log_file": str(filepath),
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

        # Update vehicle flight count
        vehicles = load_vehicles()
        for v in vehicles:
            if v["id"] == current_vehicle["id"]:
                v["flight_count"] = v.get("flight_count", 0) + 1
                current_vehicle = v
                break
        save_vehicles(vehicles)

        return JSONResponse({
            "status": "success",
            "vehicle": current_vehicle["name"],
            "drone_type": current_vehicle["drone_type_name"],
            "metrics": metrics,
            "recommendations": recommendations,
            "log_file": str(filepath)
        })

    except HTTPException:
        if filepath.exists():
            filepath.unlink()
        raise
    except Exception as e:
        if filepath.exists():
            filepath.unlink()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await file.close()


async def generate_recommendations(
    metrics: Dict,
    drone_type: str,
    notes: Optional[str] = None,
    current_params: Optional[Dict[str, float]] = None,
    model: str = "qwen2.5:7b"
) -> List[Dict]:
    """Generate parameter recommendations by communicating with Ollama JSON API."""
    recommendations: List[Dict] = []
    current_params = current_params or {}
    
    # 1. Structure the prompt to the LLM
    prompt = f"""You are an expert ArduPilot tuning AI.
The user is tuning a '{drone_type}' drone.
Flight Metrics Recorded:
{json.dumps(metrics, indent=2)}

Current parameters from the user's config:
{json.dumps(current_params, indent=2) if current_params else '{}'}

Pilot Notes/Feedback:
{notes if notes else 'None'}

Analyze the metrics and notes. 
Output your tuning recommendations strictly as a JSON list of objects with the exact schema below. Do not output anything else.

Schema:
[
  {{
    "parameter": "PARAM_NAME_A",
    "recommended": 0.15,
    "reason": "Brief technical rationale"
  }}
]

If no tuning is needed, return an empty array: []
"""

    # 2. Make Request to Local Ollama Node
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post("http://localhost:11434/api/generate", json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            })
            response.raise_for_status()
            
            data = response.json()
            llm_text = data.get("response", "[]")
            
            # Strip markdown JSON wrappers if present
            llm_text = llm_text.strip()
            if llm_text.startswith("```json"):
                llm_text = llm_text[7:]
            if llm_text.endswith("```"):
                llm_text = llm_text[:-3]
            llm_text = llm_text.strip()
            
            # 3. Parse and sanitize the response
            try:
                parsed_recs = json.loads(llm_text)
                if not isinstance(parsed_recs, list):
                    parsed_recs = []
            except json.JSONDecodeError:
                print("Failed to decode LLM JSON:", llm_text)
                parsed_recs = []
                
            for raw_rec in parsed_recs:
                parameter = raw_rec.get("parameter")
                target_value = raw_rec.get("recommended")
                reason = raw_rec.get("reason", "No reason provided")
                
                if not parameter or target_value is None:
                    continue
                    
                try:
                    target_value = float(target_value)
                except ValueError:
                    continue
                    
                add_recommendation(
                    recommendations,
                    current_params,
                    ParameterSuggestion(
                        parameter=parameter,
                        target_value=target_value,
                        reason=reason,
                    )
                )

    except Exception as e:
        print(f"Error calling Ollama API: {e}")
        # In actual production, you might raise an HTTPException, but here we'll return an empty list or mock error
        raise HTTPException(
            status_code=502,
            detail=f"Failed to communicate with local Ollama service for inference: {str(e)}"
        )

    return recommendations


def _read_history_sync(vehicle_id: Optional[int] = None) -> list:
    history = []
    for output_file in OUTPUTS_DIR.glob("*_analysis.json"):
        try:
            with open(output_file, encoding="utf-8") as f:
                data = json.load(f)
            if vehicle_id is None or data.get("vehicle_id") == vehicle_id:
                history.append({
                    "timestamp": data.get("timestamp"),
                    "vehicle_name": data.get("vehicle_name"),
                    "drone_type": data.get("drone_type"),
                    "metrics": data.get("metrics", {}),
                    "recommendations": data.get("recommendations", [])
                })
        except (OSError, json.JSONDecodeError):
            continue
    history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return history

@app.get("/history")
async def get_history(vehicle_id: Optional[int] = None):
    """Get history of all analyzed logs for a vehicle."""
    history = await asyncio.to_thread(_read_history_sync, vehicle_id)
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
async def run_training_job(job_id: str, model_name: str, epochs: int):
    """Run the actual fine-tuning script as a background subprocess."""
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "training" / "train.py"
    dataset_path = DATA_DIR / "dataset"
    output_path = TRAINING_DIR / job_id
    output_path.mkdir(parents=True, exist_ok=True)

    resolved_model = normalize_model_name(model_name)
    command = [
        "python",
        str(script_path),
        "--model",
        resolved_model,
        "--data",
        str(dataset_path),
        "--output",
        str(output_path),
        "--epochs",
        str(epochs),
    ]

    training_jobs[job_id]["status"] = "running"
    training_jobs[job_id]["resolved_model"] = resolved_model
    training_jobs[job_id]["progress"] = 2
    training_jobs[job_id]["command"] = " ".join(command)

    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(repo_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        training_processes[job_id] = process
        logs_tail: List[str] = []

        while True:
            if process.stdout is None:
                break

            line = await process.stdout.readline()
            if not line:
                break

            decoded = line.decode("utf-8", errors="ignore").strip()
            if not decoded:
                continue

            logs_tail.append(decoded)
            logs_tail = logs_tail[-50:]

            training_jobs[job_id]["latest_log"] = decoded
            training_jobs[job_id]["logs_tail"] = logs_tail

            current_progress = int(training_jobs[job_id].get("progress", 2))
            if current_progress < 95:
                training_jobs[job_id]["progress"] = current_progress + 1

        return_code = await process.wait()
        training_processes.pop(job_id, None)

        if training_jobs[job_id]["status"] == "stopped":
            return

        training_jobs[job_id]["finished_at"] = datetime.now().isoformat()
        training_jobs[job_id]["output_dir"] = str(output_path)

        if return_code == 0:
            training_jobs[job_id]["status"] = "completed"
            training_jobs[job_id]["progress"] = 100
        else:
            training_jobs[job_id]["status"] = "failed"
            training_jobs[job_id]["error"] = f"Training exited with code {return_code}"
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["error"] = str(exc)
        training_jobs[job_id]["finished_at"] = datetime.now().isoformat()
    finally:
        training_processes.pop(job_id, None)
        training_tasks.pop(job_id, None)

@app.post("/training/start")
async def start_training(
    model_name: str = Form(...),
    epochs: int = Form(3)
):
    """Start fine-tuning the model."""
    if epochs < 1 or epochs > 20:
        raise HTTPException(status_code=400, detail="Epochs must be between 1 and 20.")

    normalized_model = model_name.strip()
    if normalized_model not in MODEL_ALIASES and normalized_model not in MODEL_ALIASES.values():
        raise HTTPException(status_code=400, detail="Invalid model name. Must be a supported model.")

    if not (DATA_DIR / "dataset").exists():
        raise HTTPException(status_code=400, detail="Dataset directory is missing: data/dataset")

    job_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    training_jobs[job_id] = {
        "status": "starting",
        "model": model_name,
        "epochs": epochs,
        "started_at": datetime.now().isoformat(),
        "progress": 0
    }

    task = asyncio.create_task(run_training_job(job_id, model_name, epochs))
    training_tasks[job_id] = task

    return JSONResponse({
        "status": "started",
        "job_id": job_id,
        "message": f"Training started with {model_name} for {epochs} epochs"
    })


@app.get("/training/status")
async def get_training_status(job_id: Optional[str] = None):
    """Get training status."""
    if job_id:
        if job_id in training_jobs:
            return JSONResponse({"job_id": job_id, **training_jobs[job_id]})
        raise HTTPException(status_code=404, detail="Job not found")
    all_jobs = [{"job_id": jid, **job} for jid, job in training_jobs.items()]
    all_jobs.sort(key=lambda x: x.get("started_at", ""), reverse=True)
    return JSONResponse(all_jobs)


@app.post("/training/stop")
async def stop_training(job_id: str = Form(...)):
    """Stop a training job."""
    if job_id in training_jobs:
        process = training_processes.get(job_id)
        if process and process.returncode is None:
            process.terminate()
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=10)
            if process.returncode is None:
                process.kill()

        task = training_tasks.get(job_id)
        if task and not task.done():
            task.cancel()

        training_jobs[job_id]["status"] = "stopped"
        training_jobs[job_id]["finished_at"] = datetime.now().isoformat()
        return JSONResponse({
            "status": "stopped",
            "job_id": job_id,
            "message": "Training job stopped"
        })
    raise HTTPException(status_code=404, detail="Job not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
