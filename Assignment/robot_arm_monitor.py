"""
robot_arm_monitor.py
--------------------
Robotic Arm Joint Health Monitoring System.
Builds on Day 1 (rule engines, deterministic safety, priority) and
Day 2 (machine learning wear prediction, log scanning, signal fusion, security defense).

5 Core Modules:
1. Immediate Safety Stop (Instant, deterministic, zero-ML dependency)
2. Predictive Wear Detection (Machine learning classifier on historical data)
3. Maintenance Notes Signal (Free-text inspection log NLP scanner)
4. Spoofed-Sensor Defense (Cross-sensor thermodynamic anomaly detection)
5. Multi-Signal Decision Engine (Produces: SAFE, MONITOR, SCHEDULE_MAINTENANCE, EMERGENCY_STOP)

Course: 23IF02 System Automation
"""

import re
from pathlib import Path
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "wear_prediction_model.pkl"
MAINTENANCE_LOG_PATH = BASE_DIR / "maintenance_log.txt"

# ==============================================================
# MODULE 1: IMMEDIATE SAFETY STOP (DETERMINISTIC & ZERO-ML)
# ==============================================================
# Physical reasoning for thresholds:
# - Servomotor Class F insulation threshold is ~80°C continuous; exceeding 80°C
#   risks thermal runaway and permanent stator winding burnout.
# - ISO 10816 Zone D boundary for industrial machines is 8.5 mm/s RMS; crossing this
#   indicates imminent bearing seizure or mechanical detachment.
# - Torque limits are mechanical yield / stall limits: Shoulder (150 Nm),
#   Elbow (100 Nm), Wrist (40 Nm).
SAFETY_THRESHOLDS = {
    "temperature_max": 80.0,      # °C across all joints
    "vibration_max": 8.5,         # mm/s across all joints
    "torque_max": {
        "shoulder": 150.0,        # Nm
        "elbow": 100.0,           # Nm
        "wrist": 40.0             # Nm
    }
}

def check_immediate_safety_stop(reading):
    """
    Evaluates raw physical limits immediately and deterministically.
    Does NOT depend on any ML model, network, or external process.
    Returns: (is_emergency: bool, reason: str or None)
    """
    joints = ["shoulder", "elbow", "wrist"]
    
    # 1. Check Temperature
    for joint in joints:
        temp = reading.get(f"{joint}_temp", 0.0)
        if temp >= SAFETY_THRESHOLDS["temperature_max"]:
            return True, f"CRITICAL TEMPERATURE: {joint.upper()} temperature is {temp:.1f}°C (Exceeds limit of {SAFETY_THRESHOLDS['temperature_max']}°C)"
            
    # 2. Check Vibration
    for joint in joints:
        vib = reading.get(f"{joint}_vibration", 0.0)
        if vib >= SAFETY_THRESHOLDS["vibration_max"]:
            return True, f"DESTRUCTIVE VIBRATION: {joint.upper()} vibration is {vib:.2f} mm/s (Exceeds limit of {SAFETY_THRESHOLDS['vibration_max']} mm/s)"
            
    # 3. Check Torque
    for joint in joints:
        torque = reading.get(f"{joint}_torque", 0.0)
        limit = SAFETY_THRESHOLDS["torque_max"][joint]
        if torque >= limit:
            return True, f"MECHANICAL OVERTORQUE: {joint.upper()} torque is {torque:.1f} Nm (Exceeds limit of {limit} Nm)"
            
    return False, None


# ==============================================================
# MODULE 2: SPOOFED-SENSOR ATTACK DEFENSE
# ==============================================================
def check_sensor_spoofing(reading):
    """
    Defends against spoofed sensor attacks where an adversary or faulty sensor
    injects a fake low torque reading to mask an overheating or grinding joint.
    
    Physics Principle:
    Friction and heavy mechanical resistance dissipate energy as heat (temperature)
    and acoustic kinetic energy (vibration). If a joint is at 70°C+ and vibrating heavily,
    it is physically impossible to produce near-zero or abnormally suppressed torque
    while actively driving an industrial payload.
    
    Returns: (is_spoofed: bool, spoof_details: str or None)
    """
    joints = ["shoulder", "elbow", "wrist"]
    min_credible_torque_under_distress = {
        "shoulder": 35.0,  # Under high temp/vib, shoulder must draw >= 35 Nm
        "elbow": 20.0,     # Elbow must draw >= 20 Nm
        "wrist": 6.0       # Wrist must draw >= 6 Nm
    }
    
    for joint in joints:
        torque = reading.get(f"{joint}_torque", 0.0)
        temp = reading.get(f"{joint}_temp", 0.0)
        vib = reading.get(f"{joint}_vibration", 0.0)
        
        # Attack pattern: High temperature or high vibration, but torque is faked low
        is_distressed = (temp >= 67.0) or (vib >= 5.2)
        is_torque_suspiciously_low = torque < min_credible_torque_under_distress[joint]
        
        if is_distressed and is_torque_suspiciously_low:
            return True, (
                f"SPOOF ATTACK DETECTED on {joint.upper()}! Reported torque is {torque:.1f} Nm "
                f"(abnormally low/masked) despite severe distress (Temp: {temp:.1f}°C, Vib: {vib:.2f} mm/s). "
                f"Physical correlation violation!"
            )
            
    return False, None


# ==============================================================
# MODULE 3: PREDICTIVE WEAR DETECTION (MACHINE LEARNING)
# ==============================================================
class WearPredictor:
    """Predicts mechanical wear using the trained Random Forest model."""
    
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.features = None
        self.classes = None
        self._load_model()
        
    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Trained model not found at {self.model_path}. Run train_wear_model.py first.")
        package = joblib.load(self.model_path)
        self.model = package["model"]
        self.features = package["features"]
        self.classes = package["classes"]
        
    def predict(self, reading):
        """
        Returns: (predicted_state: str, confidence_dict: dict)
        """
        # Create single-row DataFrame with expected feature order
        df_input = pd.DataFrame([{feat: reading.get(feat, 0.0) for feat in self.features}])
        pred_state = self.model.predict(df_input)[0]
        probabilities = self.model.predict_proba(df_input)[0]
        
        confidences = {cls: round(float(prob), 4) for cls, prob in zip(self.classes, probabilities)}
        return pred_state, confidences


# ==============================================================
# MODULE 4: MAINTENANCE LOG FREE-TEXT SCANNER
# ==============================================================
CRITICAL_KEYWORDS = [
    r"grinding", r"metal shaving", r"burning", r"smoke",
    r"excessive backlash", r"jerky", r"overhaul recommended", r"seize"
]

WARNING_KEYWORDS = [
    r"clicking", r"warm to the touch", r"grease low", r"vibration",
    r"noise", r"slight backlash", r"alignment check", r"lubrication"
]

def scan_maintenance_notes(arm_id, log_file_path=MAINTENANCE_LOG_PATH):
    """
    Parses unstructured technician log entries for a given robotic arm.
    Filters out irrelevant entries (tea breaks, floor sweeping, cable ties, etc.)
    Categorizes the signal as:
    - CRITICAL_NOTE: Technician noted severe physical warning signs.
    - WARNING_NOTE: Technician noted mild/early warning signs.
    - CLEAN_NOTE: No adverse notes found for this arm.
    
    Returns: (signal_level: str, relevant_notes: list)
    """
    if not log_file_path.exists():
        return "CLEAN_NOTE", ["No maintenance log file found."]
        
    relevant_notes = []
    highest_severity = "CLEAN_NOTE"
    
    with open(log_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue
                
            # Check if this note applies to the given arm or all arms
            if arm_id.upper() in line_str.upper() or "ALL_ARMS" in line_str.upper():
                relevant_notes.append(line_str)
                
                # Check for critical keywords
                for kw in CRITICAL_KEYWORDS:
                    if re.search(kw, line_str, re.IGNORECASE):
                        highest_severity = "CRITICAL_NOTE"
                        break
                        
                # Check for warning keywords if not already critical
                if highest_severity != "CRITICAL_NOTE":
                    for kw in WARNING_KEYWORDS:
                        if re.search(kw, line_str, re.IGNORECASE):
                            highest_severity = "WARNING_NOTE"
                            break
                            
    return highest_severity, relevant_notes


# ==============================================================
# MODULE 5: MULTI-SIGNAL DECISION ENGINE
# ==============================================================
def evaluate_arm_health(arm_id, sensor_reading, predictor=None, log_path=MAINTENANCE_LOG_PATH):
    """
    Main evaluation pipeline executing all 5 project requirements.
    
    Returns a structured decision dictionary containing:
    - arm_id
    - immediate_safety_stop (bool, reason)
    - spoof_defense (bool, details)
    - ml_wear_prediction (state, confidences)
    - maintenance_log_signal (level, notes)
    - final_decision (SAFE, MONITOR, SCHEDULE_MAINTENANCE, EMERGENCY_STOP)
    - rationale (str)
    """
    result = {
        "arm_id": arm_id,
        "sensor_reading": sensor_reading,
        "immediate_safety_stop": False,
        "immediate_stop_reason": None,
        "spoof_detected": False,
        "spoof_details": None,
        "ml_wear_state": "UNKNOWN",
        "ml_confidences": {},
        "log_signal": "CLEAN_NOTE",
        "relevant_notes": [],
        "final_decision": "SAFE",
        "rationale": ""
    }
    
    # ---------------------------------------------------------
    # STEP 1: DETERMINISTIC IMMEDIATE SAFETY STOP (PRIORITY 1)
    # Instant, hard deterministic check. No ML dependency.
    # ---------------------------------------------------------
    is_emergency, emergency_reason = check_immediate_safety_stop(sensor_reading)
    if is_emergency:
        result["immediate_safety_stop"] = True
        result["immediate_stop_reason"] = emergency_reason
        result["final_decision"] = "EMERGENCY_STOP"
        result["rationale"] = f"Deterministic Safety Rule Tripped: {emergency_reason}"
        return result
        
    # ---------------------------------------------------------
    # STEP 2: SPOOFED-SENSOR DEFENSE (PRIORITY 2)
    # Defend against manipulated torque inputs.
    # ---------------------------------------------------------
    is_spoofed, spoof_details = check_sensor_spoofing(sensor_reading)
    if is_spoofed:
        result["spoof_detected"] = True
        result["spoof_details"] = spoof_details
        
    # ---------------------------------------------------------
    # STEP 3: PREDICTIVE WEAR DETECTION (MACHINE LEARNING)
    # ---------------------------------------------------------
    if predictor is not None:
        ml_state, confs = predictor.predict(sensor_reading)
        result["ml_wear_state"] = ml_state
        result["ml_confidences"] = confs
    else:
        result["ml_wear_state"] = "NORMAL"
        
    # ---------------------------------------------------------
    # STEP 4: FREE-TEXT MAINTENANCE LOG SCANNER
    # ---------------------------------------------------------
    log_signal, notes = scan_maintenance_notes(arm_id, log_path)
    result["log_signal"] = log_signal
    result["relevant_notes"] = notes
    
    # ---------------------------------------------------------
    # STEP 5: FINAL DECISION FUSION
    # Priority Order:
    # 1. EMERGENCY_STOP (Critical threshold breached)
    # 2. SCHEDULE_MAINTENANCE (Severe wear, critical notes, or spoof attack)
    # 3. MONITOR (Moderate wear or warning notes)
    # 4. SAFE (All clean and normal)
    # ---------------------------------------------------------
    
    # Condition A: Spoofed sensor attack detected
    if result["spoof_detected"]:
        # If spoofing occurred while joints are heavily stressed, trigger maintenance & security lockout
        result["final_decision"] = "SCHEDULE_MAINTENANCE"
        result["rationale"] = f"Security Defense Alert: {result['spoof_details']} Requires immediate sensor inspection and calibration."
        return result
        
    # Condition B: High wear detected via ML OR technician log
    if result["ml_wear_state"] == "SEVERE_WEAR" or result["log_signal"] == "CRITICAL_NOTE":
        result["final_decision"] = "SCHEDULE_MAINTENANCE"
        reasons = []
        if result["ml_wear_state"] == "SEVERE_WEAR":
            reasons.append("ML model predicts SEVERE_WEAR")
        if result["log_signal"] == "CRITICAL_NOTE":
            reasons.append("Technician log reported critical mechanical damage (grinding/odor/backlash)")
        result["rationale"] = " & ".join(reasons)
        return result
        
    # Condition C: Corroborating moderate wear signals (ML Moderate + Log Warning)
    if result["ml_wear_state"] == "MODERATE_WEAR" and result["log_signal"] == "WARNING_NOTE":
        result["final_decision"] = "SCHEDULE_MAINTENANCE"
        result["rationale"] = "Dual Warning: ML model predicts MODERATE_WEAR confirmed by technician observation notes. Escalated to maintenance."
        return result
        
    # Condition D: Early warning / Single moderate signal
    if result["ml_wear_state"] == "MODERATE_WEAR" or result["log_signal"] == "WARNING_NOTE":
        result["final_decision"] = "MONITOR"
        reasons = []
        if result["ml_wear_state"] == "MODERATE_WEAR":
            reasons.append("ML model detected early joint wear pattern (MODERATE_WEAR)")
        if result["log_signal"] == "WARNING_NOTE":
            reasons.append("Technician logged minor warning note (clicking/warmth)")
        result["rationale"] = " | ".join(reasons)
        return result
        
    # Condition E: Nominal healthy operation
    result["final_decision"] = "SAFE"
    result["rationale"] = "All joint parameters nominal, ML model predicts NORMAL wear, and technician logs are clean."
    return result


def print_evaluation_summary(eval_res):
    """Prints a clear, formatted summary of the evaluation."""
    arm = eval_res["arm_id"]
    dec = eval_res["final_decision"]
    
    print("-" * 70)
    print(f"ARM: {arm} | FINAL DECISION: >>> {dec} <<<")
    print("-" * 70)
    print(f"  Deterministic Safety Stop : {eval_res['immediate_safety_stop']} " +
          (f"({eval_res['immediate_stop_reason']})" if eval_res['immediate_safety_stop'] else "(Passed)"))
    print(f"  Spoofed Sensor Attack     : {eval_res['spoof_detected']} " +
          (f"({eval_res['spoof_details']})" if eval_res['spoof_detected'] else "(No spoofing)"))
    print(f"  ML Wear Model Prediction  : {eval_res['ml_wear_state']} " +
          (f"{eval_res['ml_confidences']}" if eval_res['ml_confidences'] else ""))
    print(f"  Maintenance Log Signal    : {eval_res['log_signal']}")
    if eval_res['relevant_notes']:
        print("  Technician Notes Found    :")
        for n in eval_res['relevant_notes']:
            print(f"    - {n}")
    print(f"  Decision Rationale        : {eval_res['rationale']}")
    print("-" * 70)
