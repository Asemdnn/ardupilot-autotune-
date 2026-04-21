"""
ArduPilot Parameter Definitions
Contains all tuneable parameters with their safe ranges and descriptions.
"""

import math
from typing import Optional

# ArduPilot parameter definitions organized by category
ARDUPILOT_PARAMETERS = {
    "attitude_controller": {
        "category": "Attitude Control (ATC)",
        "parameters": {
            "ATC_RAT_RLL_P": {
                "name": "Roll Rate P Gain",
                "description": "Proportional gain for roll rate controller",
                "range": [0.01, 0.5],
                "default": 0.12,
                "unit": "ratio"
            },
            "ATC_RAT_RLL_I": {
                "name": "Roll Rate I Gain",
                "description": "Integral gain for roll rate controller",
                "range": [0.01, 0.3],
                "default": 0.05,
                "unit": "ratio"
            },
            "ATC_RAT_RLL_D": {
                "name": "Roll Rate D Gain",
                "description": "Derivative gain for roll rate controller",
                "range": [0.001, 0.02],
                "default": 0.002,
                "unit": "ratio"
            },
            "ATC_RAT_RLL_IMAX": {
                "name": "Roll Rate I Max",
                "description": "Maximum integral term for roll rate",
                "range": [0.1, 1.0],
                "default": 0.5,
                "unit": "ratio"
            },
            "ATC_RAT_PIT_P": {
                "name": "Pitch Rate P Gain",
                "description": "Proportional gain for pitch rate controller",
                "range": [0.01, 0.5],
                "default": 0.12,
                "unit": "ratio"
            },
            "ATC_RAT_PIT_I": {
                "name": "Pitch Rate I Gain",
                "description": "Integral gain for pitch rate controller",
                "range": [0.01, 0.3],
                "default": 0.05,
                "unit": "ratio"
            },
            "ATC_RAT_PIT_D": {
                "name": "Pitch Rate D Gain",
                "description": "Derivative gain for pitch rate controller",
                "range": [0.001, 0.02],
                "default": 0.002,
                "unit": "ratio"
            },
            "ATC_RAT_YAW_P": {
                "name": "Yaw Rate P Gain",
                "description": "Proportional gain for yaw rate controller",
                "range": [0.01, 0.5],
                "default": 0.15,
                "unit": "ratio"
            },
            "ATC_RAT_YAW_I": {
                "name": "Yaw Rate I Gain",
                "description": "Integral gain for yaw rate controller",
                "range": [0.01, 0.3],
                "default": 0.05,
                "unit": "ratio"
            },
            "ATC_RAT_YAW_D": {
                "name": "Yaw Rate D Gain",
                "description": "Derivative gain for yaw rate controller",
                "range": [0.0, 0.02],
                "default": 0.0,
                "unit": "ratio"
            },
            "ATC_RAT_RLL_MAX": {
                "name": "Max Roll Rate",
                "description": "Maximum roll rate in deg/s",
                "range": [1800, 10800],
                "default": 7200,
                "unit": "deg/s"
            },
            "ATC_RAT_PIT_MAX": {
                "name": "Max Pitch Rate",
                "description": "Maximum pitch rate in deg/s",
                "range": [1800, 10800],
                "default": 7200,
                "unit": "deg/s"
            },
            "ATC_RAT_YAW_MAX": {
                "name": "Max Yaw Rate",
                "description": "Maximum yaw rate in deg/s",
                "range": [1800, 10800],
                "default": 3600,
                "unit": "deg/s"
            },
            # Angle controller parameters
            "ATC_ANG_RLL_P": {
                "name": "Roll Angle P Gain",
                "description": "Proportional gain for roll angle controller",
                "range": [1.0, 10.0],
                "default": 4.5,
                "unit": "ratio"
            },
            "ATC_ANG_PIT_P": {
                "name": "Pitch Angle P Gain",
                "description": "Proportional gain for pitch angle controller",
                "range": [1.0, 10.0],
                "default": 4.5,
                "unit": "ratio"
            },
            "ATC_ANG_YAW_P": {
                "name": "Yaw Angle P Gain",
                "description": "Proportional gain for yaw angle controller",
                "range": [1.0, 10.0],
                "default": 4.5,
                "unit": "ratio"
            },
        }
    },
    "position_controller": {
        "category": "Position Control (PSC)",
        "parameters": {
            "PSC_POS_Z_P": {
                "name": "Altitude P Gain",
                "description": "Proportional gain for altitude control",
                "range": [0.5, 5.0],
                "default": 1.0,
                "unit": "ratio"
            },
            "PSC_VEL_Z_P": {
                "name": "Vertical Velocity P Gain",
                "description": "Proportional gain for vertical velocity",
                "range": [1.0, 10.0],
                "default": 5.0,
                "unit": "ratio"
            },
            "PSC_POS_X_P": {
                "name": "Horizontal Position X P Gain",
                "description": "Proportional gain for X position",
                "range": [0.5, 5.0],
                "default": 1.0,
                "unit": "ratio"
            },
            "PSC_POS_Y_P": {
                "name": "Horizontal Position Y P Gain",
                "description": "Proportional gain for Y position",
                "range": [0.5, 5.0],
                "default": 1.0,
                "unit": "ratio"
            },
            "PSC_VEL_X_P": {
                "name": "Horizontal Velocity X P Gain",
                "description": "Proportional gain for X velocity",
                "range": [1.0, 10.0],
                "default": 2.0,
                "unit": "ratio"
            },
            "PSC_VEL_Y_P": {
                "name": "Horizontal Velocity Y P Gain",
                "description": "Proportional gain for Y velocity",
                "range": [1.0, 10.0],
                "default": 2.0,
                "unit": "ratio"
            },
            "WPNAV_RADIUS": {
                "name": "Waypoint Radius",
                "description": "Waypoint acceptance radius",
                "range": [0.5, 10.0],
                "default": 2.0,
                "unit": "m"
            },
        }
    },
    "filters": {
        "category": "Sensor Filters (INS)",
        "parameters": {
            "INS_GYRO_FILTER": {
                "name": "Gyro Filter Cutoff",
                "description": "Low-pass filter cutoff for gyroscope",
                "range": [20, 200],
                "default": 80,
                "unit": "Hz"
            },
            "INS_ACCEL_FILTER": {
                "name": "Accelerometer Filter Cutoff",
                "description": "Low-pass filter cutoff for accelerometer",
                "range": [5, 50],
                "default": 10,
                "unit": "Hz"
            },
            "INS_HNTC2_ENABLE": {
                "name": "Harmonic Notch Filter",
                "description": "Enable harmonic notch for vibration reduction",
                "range": [0, 1],
                "default": 0,
                "unit": "bool"
            },
            "INS_HNTC2_FREQ": {
                "name": "Harmonic Notch Frequency",
                "description": "Center frequency for harmonic notch",
                "range": [50, 500],
                "default": 120,
                "unit": "Hz"
            },
            "INS_HNTC2_BW": {
                "name": "Harmonic Notch Bandwidth",
                "description": "Bandwidth for harmonic notch filter",
                "range": [10, 200],
                "default": 50,
                "unit": "Hz"
            },
        }
    },
    "ekf": {
        "category": "Extended Kalman Filter (EK3)",
        "parameters": {
            "EK3_GPS_TYPE": {
                "name": "GPS Velocity Source",
                "description": "GPS velocity measurement type",
                "range": [0, 3],
                "default": 0,
                "unit": "enum"
            },
            "EK3_ABIAS_P": {
                "name": "Acceleration Bias Variance",
                "description": "Accelerometer bias estimation confidence",
                "range": [0.001, 0.1],
                "default": 0.01,
                "unit": "variance"
            },
            "EK3_GPS_GAIN": {
                "name": "GPS Position Weight",
                "description": "Weighting for GPS position in EKF",
                "range": [0.1, 2.0],
                "default": 1.0,
                "unit": "ratio"
            },
            "EK3_EKF_VEL_Z": {
                "name": "EKF Vertical Velocity Source",
                "description": "Source for vertical velocity estimation",
                "range": [0, 3],
                "default": 0,
                "unit": "enum"
            },
        }
    },
    "vtol": {
        "category": "VTOL Parameters",
        "parameters": {
            "VT_FW_Q_SPEED": {
                "name": "VTOL Transition Speed",
                "description": "Target airspeed during VTOL-to-fixed-wing transition",
                "range": [5, 40],
                "default": 15,
                "unit": "m/s"
            },
        }
    },
    "motor_outputs": {
        "category": "Motor Output (RCOUT)",
        "parameters": {
            "MOT_PWM_MIN": {
                "name": "Minimum PWM Output",
                "description": "Minimum PWM pulse width sent to ESCs",
                "range": [800, 1200],
                "default": 1000,
                "unit": "us"
            },
            "MOT_PWM_MAX": {
                "name": "Maximum PWM Output",
                "description": "Maximum PWM pulse width sent to ESCs",
                "range": [1500, 2200],
                "default": 2000,
                "unit": "us"
            },
            "MOT_SPIN_MIN": {
                "name": "Motor Spin Minimum",
                "description": "Minimum motor spin when armed (0=disabled)",
                "range": [0.0, 0.3],
                "default": 0.0,
                "unit": "ratio"
            },
            "MOT_SPIN_MAX": {
                "name": "Motor Spin Maximum",
                "description": "Maximum motor spin as a fraction of full throttle",
                "range": [0.9, 1.0],
                "default": 0.95,
                "unit": "ratio"
            },
            "MOT_BAT_VOLT_MAX": {
                "name": "Motor Battery Voltage Max",
                "description": "Maximum voltage for motor limit",
                "range": [12.0, 60.0],
                "default": 52.2,
                "unit": "V"
            },
            "MOT_BAT_VOLT_MIN": {
                "name": "Motor Battery Voltage Min",
                "description": "Minimum voltage for motor limit",
                "range": [6.0, 30.0],
                "default": 12.0,
                "unit": "V"
            },
        }
    },
    "flight_modes": {
        "category": "Flight Modes",
        "parameters": {
            "MODE1": {
                "name": "Flight Mode 1",
                "description": "Mode selected by channel 5 position 1",
                "range": [0, 20],
                "default": 0,
                "unit": "mode"
            },
            "MODE2": {
                "name": "Flight Mode 2",
                "description": "Mode selected by channel 5 position 2",
                "range": [0, 20],
                "default": 0,
                "unit": "mode"
            },
            "MODE3": {
                "name": "Flight Mode 3",
                "description": "Mode selected by channel 5 position 3",
                "range": [0, 20],
                "default": 0,
                "unit": "mode"
            },
            "MODE4": {
                "name": "Flight Mode 4",
                "description": "Mode selected by channel 5 position 4",
                "range": [0, 20],
                "default": 0,
                "unit": "mode"
            },
        }
    },
    "safety": {
        "category": "Safety Parameters",
        "parameters": {
            "ARMING_CHECK": {
                "name": "Arming Checks",
                "description": "Bitmask of arming checks to perform",
                "range": [0, 2147483647],
                "default": 1,
                "unit": "bitmap"
            },
            "RC_MAP_THROTTLE": {
                "name": "Throttle RC Channel",
                "description": "RC channel for throttle",
                "range": [0, 16],
                "default": 3,
                "unit": "channel"
            },
            "RC_MAP_ROLL": {
                "name": "Roll RC Channel",
                "description": "RC channel for roll",
                "range": [0, 16],
                "default": 1,
                "unit": "channel"
            },
            "RC_MAP_PITCH": {
                "name": "Pitch RC Channel",
                "description": "RC channel for pitch",
                "range": [0, 16],
                "default": 2,
                "unit": "channel"
            },
            "RC_MAP_YAW": {
                "name": "Yaw RC Channel",
                "description": "RC channel for yaw",
                "range": [0, 16],
                "default": 4,
                "unit": "channel"
            },
        }
    }
}


def validate_parameter(parameter: str, value: str) -> tuple:
    """
    Validate if a parameter value is within safe bounds.
    
    Args:
        parameter: Parameter name (e.g., "ATC_RAT_RLL_P")
        value: Proposed value as string
    
    Returns:
        Tuple of (is_valid, message)
    """
    # Search through all categories
    for category_name, category_data in ARDUPILOT_PARAMETERS.items():
        if parameter in category_data["parameters"]:
            param_info = category_data["parameters"][parameter]
            
            try:
                # Try to convert value to appropriate type
                if param_info["unit"] == "bool":
                    num_value = int(float(value))
                elif param_info["unit"] in ["enum", "mode", "channel"]:
                    num_value = int(float(value))
                elif param_info["unit"] in ["ratio", "variance", "gain"]:
                    num_value = float(value)
                else:
                    num_value = float(value)
                
                if math.isnan(num_value):
                    return False, "Value cannot be NaN"
                
                # Check if within range
                if num_value < param_info["range"][0]:
                    return False, f"Value {value} is below minimum {param_info['range'][0]}"
                if num_value > param_info["range"][1]:
                    return False, f"Value {value} exceeds maximum {param_info['range'][1]}"
                
                return True, f"Value is within safe range [{param_info['range'][0]}, {param_info['range'][1]}]"
                
            except ValueError:
                return False, f"Invalid value format: {value}"
    
    return False, f"Unknown parameter: {parameter}"


def format_recommendations(recommendations: list) -> str:
    """
    Format recommendations as a readable string.
    
    Args:
        recommendations: List of recommendation dictionaries
    
    Returns:
        Formatted string
    """
    if not recommendations:
        return "No recommendations - parameters appear optimal!"
    
    output = "Parameter Recommendations:\n"
    output += "=" * 60 + "\n\n"
    
    for i, rec in enumerate(recommendations, 1):
        output += f"{i}. {rec.get('parameter', 'Unknown')}\n"
        output += f"   Current: {rec.get('current', 'N/A')} → Recommended: {rec.get('recommended', 'N/A')}\n"
        output += f"   Change: {rec.get('change', 'N/A')}\n"
        output += f"   Reason: {rec.get('reason', 'N/A')}\n\n"
    
    return output


def get_all_parameters() -> dict:
    """Return all parameters organized by category."""
    return ARDUPILOT_PARAMETERS


def get_parameter_info(parameter: str) -> Optional[dict]:
    """Get information about a specific parameter."""
    for category_data in ARDUPILOT_PARAMETERS.values():
        if parameter in category_data["parameters"]:
            return category_data["parameters"][parameter]
    return None
