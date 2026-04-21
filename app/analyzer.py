"""
Flight Log Analyzer for ArduPilot
Parses .bin and .log files and extracts key performance metrics using pandas/numpy for memory-safe analysis.
"""

import os
import math
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from pathlib import Path

def analyze_flight_log(log_path: str) -> Dict:
    """
    Analyze an ArduPilot flight log and extract key metrics memory-safely.
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
        "loop_time_ms": 0.0,  # CLpt converted to ms (was cpu_load_max)
    }
    
    if not os.path.exists(log_path):
        return metrics

    try:
        import pymavlink
        from pymavlink import mavutil
        
        mlog = mavutil.mavlink_connection(log_path)
        
        # memory safe on-the-fly aggregations
        vibration_max = 0.0
        volt_min = float('inf')
        cpu_max = 0.0
        gps_hdop_max = 0.0
        max_roll_rate = 0.0
        max_pitch_rate = 0.0
        
        # for maneuver step response
        rate_data = {
            'TimeUS': [],
            'RDes': [],
            'R': [],
            'PDes': [],
            'P': []
        }
        
        try:
            while True:
                msg = mlog.recv_match(type=['RATE', 'GPS', 'BAT', 'BATT', 'BAT2', 'VIBE', 'PERF'], blocking=False)
                if msg is None:
                    break
                    
                msg_type = msg.get_type()
            
            if msg_type == "PERF" and hasattr(msg, 'CLpt'):
                # CLpt is loop execution time in microseconds; convert to ms
                cpu_max = max(cpu_max, msg.CLpt / 1000.0)
            
            elif msg_type == 'RATE':
                # Use getattr with defaults in case of legacy names
                time_us = getattr(msg, 'TimeUS', 0)
                r_des = getattr(msg, 'RDes', 0.0)
                r_act = getattr(msg, 'R', 0.0)
                p_des = getattr(msg, 'PDes', 0.0)
                p_act = getattr(msg, 'P', 0.0)
                
                max_roll_rate = max(max_roll_rate, abs(r_act))
                max_pitch_rate = max(max_pitch_rate, abs(p_act))
                
                rate_data['TimeUS'].append(time_us)
                rate_data['RDes'].append(r_des)
                rate_data['R'].append(r_act)
                rate_data['PDes'].append(p_des)
                rate_data['P'].append(p_act)
                
            elif msg_type == "GPS" and hasattr(msg, 'HDop'):
                gps_hdop_max = max(gps_hdop_max, msg.HDop)
                
            elif msg_type in ['BAT', 'BATT', 'BAT2'] and hasattr(msg, 'Volt'):
                volt_min = min(volt_min, msg.Volt)
                
            elif msg_type == 'VIBE':
                # ArduPilot 4.x exposes VibeX/VibeY/VibeZ natively in VIBE
                vx = getattr(msg, 'VibeX', 0.0)
                vy = getattr(msg, 'VibeY', 0.0)
                vz = getattr(msg, 'VibeZ', 0.0)
                v = math.sqrt(vx**2 + vy**2 + vz**2)
                vibration_max = max(vibration_max, v)
        finally:
            mlog.close()

        metrics['vibration_g'] = vibration_max
        metrics['battery_voltage_min'] = volt_min if volt_min != float('inf') else 0.0
        metrics['loop_time_ms'] = cpu_max  # renamed from cpu_load_max: CLpt in ms
        metrics['max_roll_rate'] = max_roll_rate
        metrics['max_pitch_rate'] = max_pitch_rate
        metrics['gps_horizontal_accuracy'] = gps_hdop_max
        
        df = pd.DataFrame(rate_data)
        if not df.empty:
            df['RollErr'] = (df['RDes'] - df['R']).abs()
            df['PitchErr'] = (df['PDes'] - df['P']).abs()
            metrics['max_roll_error'] = float(df['RollErr'].max())
            metrics['max_pitch_error'] = float(df['PitchErr'].max())
            
            # calculate exact step response using pandas
            overshoot, settling = _analyze_maneuvers(df)
            metrics['overshoot_percent'] = overshoot
            metrics['settling_time_sec'] = settling

            # compute duration
            dur_us = df['TimeUS'].iloc[-1] - df['TimeUS'].iloc[0]
            metrics['flight_duration_sec'] = float(dur_us / 1e6)
            
    except ImportError:
        # pymavlink not available
        metrics = _basic_parse(log_path)
    except Exception as e:
        print(f"Error parsing log with pymavlink: {e}")
        metrics = _basic_parse(log_path)
        
    return metrics


def _analyze_maneuvers(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Locates rapid stick maneuvers (step inputs) and calculates exact overshoot and settling times.
    """
    overshoots = []
    settling_times = []
    
    # Convert microseconds to seconds
    df['TimeS'] = df['TimeUS'] / 1e6
    
    for axis in ['R', 'P']:
        des_col = f'{axis}Des'
        act_col = axis
        
        # Calculate rate of change of desired input over time (deg/s^2)
        dt = df['TimeS'].diff()
        
        # 2500 deg/s^2 acceleration threshold identifies a hard stick input
        step_threshold = 2500.0  
        
        df['DesDiff'] = df[des_col].diff().abs()
        dt_safe = np.where(dt > 0.001, dt, 1.0) # avoid division by zero or NaN
        df['DesAcc'] = np.where(dt > 0.001, df['DesDiff'] / dt_safe, 0.0)
        
        step_indices = df.index[df['DesAcc'] > step_threshold].tolist()
        
        # Cluster steps to prevent capturing the same maneuver continuously
        cleaned_steps = []
        last_t = -100.0
        for idx in step_indices:
            t_curr = df.at[idx, 'TimeS']
            if t_curr - last_t > 2.0:  # Only one maneuver per 2 seconds
                cleaned_steps.append(idx)
                last_t = t_curr
            
        for step_idx in cleaned_steps:
            t0 = df.at[step_idx, 'TimeS']
            
            # Analyze a 2-second window after the maneuver begins
            window = df[(df['TimeS'] >= t0) & (df['TimeS'] <= t0 + 2.0)]
            if window.empty:
                continue
                
            target_value = window[des_col].iloc[-1]
            if abs(target_value) < 10: # Ignore tiny maneuvers
                continue
                
            # Find the peak actual rate in the direction of the target
            if target_value > 0:
                max_act = window[act_col].max()
            else:
                max_act = window[act_col].min()
            
            # Calculate overshoot
            overshoot_val = ((abs(max_act) - abs(target_value)) / abs(target_value)) * 100
            overshoot_val = max(0.0, min(overshoot_val, 100.0))
            overshoots.append(overshoot_val)
            
            # Calculate settling time (time to stay within 5% of target)
            tolerance = max(abs(target_value) * 0.05, 5.0)  # 5% or 5 deg/s

            # Track the start of the *last* contiguous band entry.
            # If the response rings in and out, we reset and only count the
            # final unbroken stint inside the tolerance band.
            settled_time = None
            current_entry = None

            for _, row in window.iterrows():
                val = row[act_col]
                if abs(val - target_value) <= tolerance:
                    if current_entry is None:
                        current_entry = row['TimeS'] - t0
                else:
                    # Left the band — reset
                    if current_entry is not None:
                        settled_time = current_entry
                    current_entry = None

            # If still inside the band at end of window, record that entry
            if current_entry is not None:
                settled_time = current_entry

            settling_times.append(settled_time if settled_time is not None else 2.0)
                
    final_overshoot = float(np.mean(overshoots)) if overshoots else 0.0
    final_settling = float(np.mean(settling_times)) if settling_times else 0.0
    
    return final_overshoot, final_settling


def _basic_parse(log_path: str) -> Dict:
    """
    Graceful fallback for files that unwrap without pymavlink.
    Prevents reading binary streams as strings.
    """
    return {
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
        "loop_time_ms": 0.0,
    }

def extract_key_metrics(log_path: str) -> Dict:
    return analyze_flight_log(log_path)

def get_log_info(log_path: str) -> Dict:
    info = {
        "filename": Path(log_path).name,
        "size_bytes": os.path.getsize(log_path) if os.path.exists(log_path) else 0,
    }
    return info

def parse_log(log_path: str) -> Dict:
    return analyze_flight_log(log_path)
