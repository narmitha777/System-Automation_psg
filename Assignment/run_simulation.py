"""
run_simulation.py
-----------------
Demonstration runner for the Robotic Arm Joint Health Monitoring System.
Explicitly demonstrates that all 4 possible final decisions are reachable:
1. SAFE
2. MONITOR
3. SCHEDULE_MAINTENANCE
4. EMERGENCY_STOP
Plus a 5th scenario demonstrating the Spoofed-Sensor Attack Defense in action!

Saves complete execution trace to simulation_output.log.

Course: 23IF02 System Automation
"""

import sys
from pathlib import Path
from robot_arm_monitor import WearPredictor, evaluate_arm_health, print_evaluation_summary

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "simulation_output.log"

class TeeLogger:
    """Tee logger to output both to terminal and log file."""
    def __init__(self, terminal, file_path):
        self.terminal = terminal
        self.file = open(file_path, "w", encoding="utf-8")
        
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()
        
    def flush(self):
        self.terminal.flush()
        self.file.flush()
        
    def close(self):
        self.file.close()

def main():
    original_stdout = sys.stdout
    tee = TeeLogger(original_stdout, LOG_FILE)
    sys.stdout = tee
    
    try:
        print("=" * 75)
        print("     ROBOTIC ARM JOINT HEALTH MONITORING SYSTEM - SIMULATION RUN     ")
        print("         23IF02 System Automation | Bonus Challenge Submission        ")
        print("=" * 75)
        print(f"Log destination: {LOG_FILE.name}\n")
        
        # Load the predictive wear model
        predictor = WearPredictor()
        print("[OK] Loaded predictive machine learning model (RandomForestClassifier)\n")
        
        scenarios = [
            # -------------------------------------------------------------
            # SCENARIO 1: Expected Outcome -> SAFE
            # -------------------------------------------------------------
            {
                "description": "SCENARIO 1: Healthy Arm under Nominal Production",
                "arm_id": "ARM_06",
                "reading": {
                    "shoulder_torque": 52.0, "shoulder_temp": 44.0, "shoulder_vibration": 1.8,
                    "elbow_torque": 32.0,    "elbow_temp": 38.0,    "elbow_vibration": 1.5,
                    "wrist_torque": 12.0,    "wrist_temp": 34.0,    "wrist_vibration": 1.2
                }
            },
            
            # -------------------------------------------------------------
            # SCENARIO 2: Expected Outcome -> MONITOR
            # Technician log has minor warning note (clicking/grease low),
            # while sensor telemetry is still within acceptable bounds.
            # -------------------------------------------------------------
            {
                "description": "SCENARIO 2: Early Warning / Observation (Technician Notice)",
                "arm_id": "ARM_02",
                "reading": {
                    "shoulder_torque": 55.0, "shoulder_temp": 46.0, "shoulder_vibration": 1.9,
                    "elbow_torque": 35.0,    "elbow_temp": 41.0,    "elbow_vibration": 2.1,
                    "wrist_torque": 13.0,    "wrist_temp": 36.0,    "wrist_vibration": 1.4
                }
            },
            
            # -------------------------------------------------------------
            # SCENARIO 3: Expected Outcome -> SCHEDULE_MAINTENANCE
            # Severe joint wear detected by ML and confirmed by technician log
            # -------------------------------------------------------------
            {
                "description": "SCENARIO 3: Severe Mechanical Wear & Critical Grinding Notes",
                "arm_id": "ARM_03",
                "reading": {
                    "shoulder_torque": 126.0, "shoulder_temp": 74.0, "shoulder_vibration": 6.9,
                    "elbow_torque": 82.0,     "elbow_temp": 68.0,    "elbow_vibration": 5.8,
                    "wrist_torque": 32.0,     "wrist_temp": 62.0,    "wrist_vibration": 5.2
                }
            },
            
            # -------------------------------------------------------------
            # SCENARIO 4: Expected Outcome -> EMERGENCY_STOP
            # Deterministic safety rule tripped: Instant stop without ML
            # -------------------------------------------------------------
            {
                "description": "SCENARIO 4: Extreme Thermal/Mechanical Hazard (Instant Safety Stop)",
                "arm_id": "ARM_01",
                "reading": {
                    "shoulder_torque": 158.0, "shoulder_temp": 88.5, "shoulder_vibration": 9.4,
                    "elbow_torque": 65.0,     "elbow_temp": 58.0,    "elbow_vibration": 3.2,
                    "wrist_torque": 18.0,     "wrist_temp": 45.0,    "wrist_vibration": 2.1
                }
            },
            
            # -------------------------------------------------------------
            # SCENARIO 5: Expected Outcome -> SCHEDULE_MAINTENANCE (SPOOF ATTACK)
            # Adversary spoofs torque to fake 10.0 Nm while joint is overheating
            # -------------------------------------------------------------
            {
                "description": "SCENARIO 5: Spoofed-Sensor Attack Defense in Action",
                "arm_id": "ARM_05",
                "reading": {
                    "shoulder_torque": 11.5, "shoulder_temp": 75.0, "shoulder_vibration": 6.8,  # Fake torque!
                    "elbow_torque": 78.0,    "elbow_temp": 66.0,    "elbow_vibration": 5.4,
                    "wrist_torque": 29.0,    "wrist_temp": 60.0,    "wrist_vibration": 4.8
                }
            }
        ]
        
        outcomes_seen = set()
        
        for idx, sc in enumerate(scenarios, start=1):
            print("\n" + "#" * 75)
            print(f"TEST CASE {idx}: {sc['description']}")
            print("#" * 75)
            
            res = evaluate_arm_health(
                arm_id=sc["arm_id"],
                sensor_reading=sc["reading"],
                predictor=predictor
            )
            
            print_evaluation_summary(res)
            outcomes_seen.add(res["final_decision"])
            
        print("\n" + "=" * 75)
        print("                     SIMULATION VERIFICATION SUMMARY                  ")
        print("=" * 75)
        expected_outcomes = {"SAFE", "MONITOR", "SCHEDULE_MAINTENANCE", "EMERGENCY_STOP"}
        print(f"Total test cases executed    : {len(scenarios)}")
        print(f"Unique decisions reached     : {sorted(list(outcomes_seen))}")
        print(f"All 4 mandatory outcomes met : {expected_outcomes.issubset(outcomes_seen)}")
        
        if expected_outcomes.issubset(outcomes_seen):
            print("\n>>> ALL ASSIGNMENT REQUIREMENTS VERIFIED AND REACHABLE SUCCESSFULLY! <<<")
        else:
            missing = expected_outcomes - outcomes_seen
            print(f"\n[WARNING] Missing expected outcomes: {missing}")
            
    finally:
        sys.stdout = original_stdout
        tee.close()
        print(f"\nSimulation completed. Full log written to: {LOG_FILE}")

if __name__ == "__main__":
    main()
