"""
generate_dataset.py
-------------------
Synthetic Dataset and Maintenance Log Generator for Robotic Arm Joint Health.

Generates:
1. synthetic_robot_arm_data.csv - At least 50 (here 120) labeled historical records
   of joint torque, temperature, and vibration across shoulder, elbow, and wrist joints.
   Labels: NORMAL, MODERATE_WEAR, SEVERE_WEAR.
2. maintenance_log.txt - Technician inspection logs in free text with at least 8
   (here 12) entries mixing relevant mechanical observations and irrelevant notes.

Course: 23IF02 System Automation
"""

import random
import pandas as pd
from pathlib import Path

# Target directory
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "synthetic_robot_arm_data.csv"
LOG_PATH = BASE_DIR / "maintenance_log.txt"

random.seed(42)

def generate_synthetic_dataset(num_samples=120):
    """
    Generate realistic joint readings for a 6-axis robotic arm fleet.
    Monitors shoulder, elbow, and wrist joints across 3 health states:
    - NORMAL (healthy operating range)
    - MODERATE_WEAR (early stage degradation, increased friction/vibration)
    - SEVERE_WEAR (imminent failure, elevated thermal dissipation and rough vibration)
    """
    records = []
    
    # 50 Normal, 40 Moderate Wear, 30 Severe Wear
    states = ["NORMAL"] * 50 + ["MODERATE_WEAR"] * 40 + ["SEVERE_WEAR"] * 30
    random.shuffle(states)
    
    for i, state in enumerate(states, start=1):
        arm_id = f"ARM_{random.randint(1, 6):02d}"
        
        if state == "NORMAL":
            shoulder_torque = round(random.uniform(45.0, 85.0), 2)
            shoulder_temp = round(random.uniform(38.0, 58.0), 2)
            shoulder_vib = round(random.uniform(1.0, 3.2), 2)
            
            elbow_torque = round(random.uniform(25.0, 55.0), 2)
            elbow_temp = round(random.uniform(35.0, 52.0), 2)
            elbow_vib = round(random.uniform(0.9, 3.0), 2)
            
            wrist_torque = round(random.uniform(8.0, 20.0), 2)
            wrist_temp = round(random.uniform(30.0, 48.0), 2)
            wrist_vib = round(random.uniform(0.8, 2.5), 2)
            
        elif state == "MODERATE_WEAR":
            # 1 or 2 joints show signs of wear
            affected = random.choice(["shoulder", "elbow", "wrist", "both"])
            
            shoulder_torque = round(random.uniform(85.0, 108.0) if affected in ["shoulder", "both"] else random.uniform(50.0, 80.0), 2)
            shoulder_temp = round(random.uniform(58.0, 67.0) if affected in ["shoulder", "both"] else random.uniform(42.0, 58.0), 2)
            shoulder_vib = round(random.uniform(3.5, 5.2) if affected in ["shoulder", "both"] else random.uniform(1.2, 3.2), 2)
            
            elbow_torque = round(random.uniform(55.0, 74.0) if affected in ["elbow", "both"] else random.uniform(28.0, 52.0), 2)
            elbow_temp = round(random.uniform(52.0, 63.0) if affected in ["elbow", "both"] else random.uniform(36.0, 52.0), 2)
            elbow_vib = round(random.uniform(3.2, 4.8) if affected in ["elbow", "both"] else random.uniform(1.0, 2.9), 2)
            
            wrist_torque = round(random.uniform(20.0, 27.0) if affected == "wrist" else random.uniform(9.0, 19.0), 2)
            wrist_temp = round(random.uniform(48.0, 57.0) if affected == "wrist" else random.uniform(32.0, 47.0), 2)
            wrist_vib = round(random.uniform(2.8, 4.2) if affected == "wrist" else random.uniform(0.9, 2.4), 2)
            
        else:  # SEVERE_WEAR
            # Severe wear on one or more joints
            shoulder_torque = round(random.uniform(110.0, 142.0), 2)
            shoulder_temp = round(random.uniform(68.0, 78.5), 2)
            shoulder_vib = round(random.uniform(5.5, 8.1), 2)
            
            elbow_torque = round(random.uniform(75.0, 94.0), 2)
            elbow_temp = round(random.uniform(64.0, 76.0), 2)
            elbow_vib = round(random.uniform(5.0, 7.6), 2)
            
            wrist_torque = round(random.uniform(28.0, 37.0), 2)
            wrist_temp = round(random.uniform(58.0, 73.0), 2)
            wrist_vib = round(random.uniform(4.5, 7.4), 2)
            
        records.append({
            "sample_id": f"SMP_{i:03d}",
            "arm_id": arm_id,
            "shoulder_torque": shoulder_torque,
            "shoulder_temp": shoulder_temp,
            "shoulder_vibration": shoulder_vib,
            "elbow_torque": elbow_torque,
            "elbow_temp": elbow_temp,
            "elbow_vibration": elbow_vib,
            "wrist_torque": wrist_torque,
            "wrist_temp": wrist_temp,
            "wrist_vibration": wrist_vib,
            "wear_status": state
        })
        
    df = pd.DataFrame(records)
    df.to_csv(DATASET_PATH, index=False)
    print(f"[OK] Generated {len(df)} synthetic records -> {DATASET_PATH.name}")
    print("Class distribution:")
    print(df["wear_status"].value_counts())
    return df

def generate_sample_maintenance_log():
    """
    Generate at least 8 (here 12) realistic technician maintenance notes.
    Mixes relevant mechanical observations with irrelevant facility/routine entries.
    """
    entries = [
        "2026-10-10 08:30:15 | ARM_01 | Routine shift inspection completed. Wiped dust shield and verified optical light curtain.",
        "2026-10-10 11:15:42 | ARM_03 | Technician noticed audible metallic grinding noise coming from shoulder bearing during rapid traverse.",
        "2026-10-11 09:05:00 | ARM_02 | Minor rhythmic clicking sound detected on elbow joint under 4kg payload. Grease level low.",
        "2026-10-11 14:20:18 | ARM_04 | Cleaned external pneumatic tubing and replaced quick-disconnect clip on tool head.",
        "2026-10-12 10:00:00 | FACILITY | Morning maintenance tea break taken. Spare parts inventory updated in warehouse kiosk.",
        "2026-10-12 16:45:30 | ARM_05 | Excessive backlash and jerky articulation observed on elbow joint. Needs mechanical alignment check.",
        "2026-10-13 08:12:10 | ARM_06 | All joints running smoothly. Thermal scan confirms normal baseline operating temperatures.",
        "2026-10-13 13:50:22 | ARM_01 | Faint burning electrical odor detected around wrist servomotor casing after 3-hour continuous run.",
        "2026-10-14 09:30:45 | ARM_03 | Visible metal shavings found on shoulder joint mounting flange. Immediate overhaul recommended.",
        "2026-10-14 15:10:00 | CELL_4 | Re-calibrated area safety scanner and swept work cell floor.",
        "2026-10-15 11:25:33 | ARM_04 | Wrist joint casing slightly warm to the touch, but movement remains smooth and within spec.",
        "2026-10-15 17:00:10 | ARM_02 | Replaced barcode scanner cable tie. Operator reported no functional anomalies."
    ]
    
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(entry + "\n")
            
    print(f"[OK] Generated {len(entries)} maintenance log entries -> {LOG_PATH.name}")

if __name__ == "__main__":
    generate_synthetic_dataset(120)
    generate_sample_maintenance_log()
