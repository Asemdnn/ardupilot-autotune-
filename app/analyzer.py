"""
Flight Log Analyzer for ArduPilot
Parses .bin and .log files and extracts key performance metrics.
"""

import os
from typing import Dict, Optional, List
from pathlib import Path


def analyze_flight_log(log_path: str) -> Dict:
    """
    Analyze an ArduPilot flight log and extract key metrics.
    
    Args:
        log_path: Path to the .bin or .log file
    
    Returns:
        Dictionary of extracted metrics
    """
    metrics = {
        "max_roll_error": 0.0,
        "max_pitch_error": 0.0,
        "overshoot_percent": 0.0,
        "settling_time_sec": 0.0,
        "vibration_g": 0.0,
        "max_roll_rate": 0.0,
        "max_pitch_rate": 0.0,
        "flight_duration_sec": 0.0,
        "num_waypoints": 0,
        "gps_horizontal_accuracy": 0.0,
        "battery_voltage_min": 0.0,
        "cpu_load_max": 0.0,
    }
    
    # Check if file exists
    if not os.path.exists(log_path):
        return metrics
    
    # Try to parse with pymavlink if available
    try:
        import pymavlink
        from pymavlink import mavutil
        
        # Try to read the log
        mlog = mavutil.mavlink_connection(log_path)
        
        # Initialize data containers
        roll_errors = []
        pitch_errors = []
        roll_rates = []
        pitch_rates = []
        vibrations = []
        voltages = []
        cpu_loads = []
        
        # Parse messages
        while True:
            msg = mlog.recv_match(blocking=False)
            if msg is None:
                break
            
            msg_type = msg.get_type()
            
            # Extract relevant data from different message types
            if msg_type == "ERR":
                if hasattr(msg, 'Subsystem'):
                    # Error logging
                    pass
            
            elif msg_type == "PERF":
                if hasattr(msg, 'CLpt'):
                    cpu_loads.append(msg.CLpt)
            
            # Check for ATT (attitude) messages
            if msg_type == "ATT":
                if hasattr(msg, 'Roll'):
                    roll_rates.append(abs(msg.Roll))
                if hasattr(msg, 'Pitch'):
                    pitch_rates.append(abs(msg.Pitch))
            
            # Check for rate controller data
            if msg_type in ['RATE', 'RCLL', 'RCRL']:
                if hasattr(msg, 'Des'):
                    roll_errors.append(abs(msg.Des))
            
            # Check for GPS data
            if msg_type == "GPS":
                if hasattr(msg, 'HDop'):
                    metrics['gps_horizontal_accuracy'] = msg.HDop
            
            # Check for battery data
            if msg_type in ['BATT', 'BAT2']:
                if hasattr(msg, 'Volt'):
                    voltages.append(msg.Volt)
            
            # Check for vibration data
            if msg_type in ['IMU', 'IMU2', 'IMU3']:
                if hasattr(msg, 'Vibe'):
                    vibrations.append(sum(msg.Vibe) / 3 if isinstance(msg.Vibe, (list, tuple)) else msg.Vibe)
        
        # Calculate statistics
        if roll_errors:
            metrics['max_roll_error'] = max(roll_errors)
        if pitch_errors:
            metrics['max_pitch_error'] = max(pitch_errors)
        if roll_rates:
            metrics['max_roll_rate'] = max(roll_rates)
        if pitch_rates:
            metrics['max_pitch_rate'] = max(pitch_rates)
        if vibrations:
            metrics['vibration_g'] = max(vibrations)
        if voltages:
            metrics['battery_voltage_min'] = min(voltages)
        if cpu_loads:
            metrics['cpu_load_max'] = max(cpu_loads)
            
    except ImportError:
        # pymavlink not available, use basic parsing
        metrics = _basic_parse(log_path)
    except Exception as e:
        # Fall back to basic parsing on any error
        print(f"Error parsing log with pymavlink: {e}")
        metrics = _basic_parse(log_path)
    
    # Calculate derived metrics
    metrics['overshoot_percent'] = _calculate_overshoot(metrics)
    metrics['settling_time_sec'] = _estimate_settling_time(metrics)
    
    return metrics


def _basic_parse(log_path: str) -> Dict:
    """
    Basic text-based parsing for log files when pymavlink is unavailable.
    """
    metrics = {
        "max_roll_error": 0.0,
        "max_pitch_error": 0.0,
        "overshoot_percent": 0.0,
        "settling_time_sec": 0.0,
        "vibration_g": 0.0,
        "max_roll_rate": 0.0,
        "max_pitch_rate": 0.0,
        "flight_duration_sec": 0.0,
    }
    
    try:
        with open(log_path, 'r', errors='ignore') as f:
            content = f.read(100000)  # Read first 100KB
            
            # Simple keyword matching
            import re
            
            # Look for roll errors
            roll_matches = re.findall(r'Roll.*?(\d+\.?\d*)', content)
            if roll_matches:
                metrics['max_roll_error'] = max(float(m) for m in roll_matches[:100])
            
            # Look for pitch errors
            pitch_matches = re.findall(r'Pitch.*?(\d+\.?\d*)', content)
            if pitch_matches:
                metrics['max_pitch_error'] = max(float(m) for m in pitch_matches[:100])
            
            # Look for vibration
            vibe_matches = re.findall(r'Vibe.*?(\d+\.?\d*)', content)
            if vibe_matches:
                metrics['vibration_g'] = max(float(m) for m in vibe_matches[:100])
                
    except Exception as e:
        print(f"Basic parse error: {e}")
    
    return metrics


def _calculate_overshoot(metrics: Dict) -> float:
    """
    Estimate overshoot percentage from error metrics.
    """
    max_error = max(metrics.get('max_roll_error', 0), metrics.get('max_pitch_error', 0))
    
    if max_error < 5:
        return 0.0
    elif max_error < 15:
        return (max_error - 5) * 0.5
    else:
        return min((max_error - 5) * 0.8, 30)


def _estimate_settling_time(metrics: Dict) -> float:
    """
    Estimate settling time based on controller performance.
    """
    overshoot = metrics.get('overshoot_percent', 0)
    vibration = metrics.get('vibration_g', 0)
    
    base_time = 1.0
    
    if overshoot > 10:
        base_time += 0.5
    if vibration > 2.0:
        base_time += 0.3
    
    return base_time


def extract_key_metrics(log_path: str) -> Dict:
    """
    Lightweight metric extraction for quick analysis.
    """
    return analyze_flight_log(log_path)


def get_log_info(log_path: str) -> Dict:
    """
    Get basic information about a flight log.
    """
    info = {
        "filename": Path(log_path).name,
        "size_bytes": os.path.getsize(log_path) if os.path.exists(log_path) else 0,
    }
    
    return info


# Legacy function for compatibility
def parse_log(log_path: str) -> Dict:
    """Legacy function name."""
    return analyze_flight_log(log_path)
