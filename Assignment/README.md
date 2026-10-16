# Robotic Arm Joint Health Monitoring System

**Course:** 23IF02 System Automation  
**Challenge:** Bonus Challenge Submission  

This project implements an end-to-end automated health monitoring system for a fleet of 6-axis robotic arms on an assembly line. It builds directly upon the concepts developed in **Day 1** (deterministic rule engines, priority arbitration) and **Day 2** (predictive machine learning, log parsing, multi-signal fusion, security validation).

---

## Project Structure

```
Assignment/
├── generate_dataset.py          # Generates synthetic sensor dataset & technician log
├── synthetic_robot_arm_data.csv # 120 labeled historical joint telemetry records
├── maintenance_log.txt          # Sample technician inspection notes (relevant & irrelevant)
├── train_wear_model.py          # Trains and evaluates Random Forest wear classifier
├── wear_prediction_model.pkl    # Serialized trained model & metadata
├── robot_arm_monitor.py         # Full monitoring pipeline implementing all 5 requirements
├── run_simulation.py            # Simulation runner testing all 4 decisions + spoof defense
├── simulation_output.log        # Complete terminal output trace from simulation run
├── EXPLANATION.md               # Technical explanation covering thresholds, fusion, & spoofing
└── README.md                    # This documentation file
```

---

## The 5 Core Requirements Implemented

1. **Immediate Safety Stop (Requirement 1):**  
   Deterministic, zero-latency safety interlock operating purely on raw physical thresholds (Temperature $\ge 80^\circ\text{C}$, Vibration $\ge 8.5\text{ mm/s}$, Shoulder Torque $\ge 150\text{ Nm}$, Elbow $\ge 100\text{ Nm}$, Wrist $\ge 40\text{ Nm}$). Guaranteed to work without any dependency on the ML model.

2. **Predictive Wear Detection (Requirement 2):**  
   Predicts joint degradation before safety hazards occur using a Random Forest Classifier trained on 120 multi-joint records (`NORMAL`, `MODERATE_WEAR`, `SEVERE_WEAR`).

3. **Maintenance Notes Signal (Requirement 3):**  
   Independent text scanner that parses free-form technician inspection notes from `maintenance_log.txt`, distinguishes critical damage (grinding, metal shavings) and early warnings (clicking, warmth) from irrelevant routine entries (cleaning, tea breaks).

4. **Spoofed-Sensor Defense (Requirement 4):**  
   Cross-sensor thermodynamic anomaly detector that detects when an attacker or failing transducer injects a fake low torque value to mask an overheating or heavily vibrating joint.

5. **Multi-Signal Decision Fusion (Requirement 5):**  
   Synthesizes all signals to produce exactly one clear final decision per robotic arm:
   - `SAFE`
   - `MONITOR`
   - `SCHEDULE_MAINTENANCE`
   - `EMERGENCY_STOP`

---

## How to Run

### Step 1: Generate Dataset and Maintenance Log
```bash
python generate_dataset.py
```
Outputs `synthetic_robot_arm_data.csv` (120 rows) and `maintenance_log.txt` (12 entries).

### Step 2: Train the Predictive Model
```bash
python train_wear_model.py
```
Trains the Random Forest model and saves it to `wear_prediction_model.pkl`.

### Step 3: Run the Complete Simulation
```bash
python run_simulation.py
```
Runs 5 distinct scenarios, proving that all 4 final decisions are reachable and demonstrating the spoof attack defense. Full log is saved to `simulation_output.log`.

---

## Verification of Four Reachable Decisions

The simulation run verifies that each outcome is reachable:
- **Scenario 1 (ARM_06):** Nominal parameters, clean log -> **`SAFE`**
- **Scenario 2 (ARM_02):** Normal sensors, minor clicking noted in log -> **`MONITOR`**
- **Scenario 3 (ARM_03):** Severe wear telemetry + metal shavings in log -> **`SCHEDULE_MAINTENANCE`**
- **Scenario 4 (ARM_01):** Temperature $88.5^\circ\text{C}$ exceeds emergency limit -> **`EMERGENCY_STOP`**
- **Scenario 5 (ARM_05):** Spoofed torque masking high heat & vibration -> **`SCHEDULE_MAINTENANCE`** (Spoof Alert)
