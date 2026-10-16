# Robotic Arm Joint Health Monitoring System: Technical Explanation

**Course:** 23IF02 System Automation  
**Challenge:** Bonus Challenge — Robotic Arm Joint Health Monitoring  

---

### 1. Why Specific Safety Thresholds Were Chosen for the Immediate-Stop Check

The immediate safety-stop check is designed to operate instantly and deterministically as a hard-coded, zero-latency safety interlock. It evaluates raw sensor readings on every control cycle and never waits for machine learning inference, network communication, or multi-sample aggregation. This guarantees fail-safe operation even if the ML model crashes, predicts an incorrect state, or is completely offline.

The thresholds were chosen based on standard electromechanical limits and ISO industrial vibration standards for 6-axis articulated assembly robots:

1. **Joint Temperature Limit ($80.0^\circ\text{C}$ across all joints):**  
   Industrial servomotors typically utilize Class F winding insulation rated for $155^\circ\text{C}$ absolute maximum, but joint housing and harmonic gear grease (such as synthetic fluorinated lubricants) degrade rapidly above $80^\circ\text{C}$. Operating beyond $80^\circ\text{C}$ causes rapid lubricant breakdown, seal melting, and permanent demagnetization of rare-earth permanent magnets in the servomotor. Thus, $80^\circ\text{C}$ serves as the non-negotiable emergency thermal cutoff.

2. **Vibration Severity Limit ($8.5\text{ mm/s RMS}$ across all joints):**  
   In accordance with **ISO 10816** (Evaluation of machine vibration on non-rotating parts) Zone D criteria, vibration exceeding $7.1\text{ mm/s}$ to $8.5\text{ mm/s}$ is classified as *unacceptable and destructive*. At this amplitude, joint bearings suffer severe raceway spalling, gear tooth micro-fractures occur, and there is immediate danger of structural linkage detachment or collision. An instant trip at $8.5\text{ mm/s}$ protects plant personnel and robotic hardware from catastrophic failure.

3. **Maximum Joint Torque Limits (Shoulder: $150\text{ Nm}$, Elbow: $100\text{ Nm}$, Wrist: $40\text{ Nm}$):**  
   The mechanical yield limits reflect the geometric and kinematic load distribution of a 6-axis serial manipulator:
   - The **shoulder joint** bears the mass of the entire arm plus payload; $150\text{ Nm}$ is set just below the motor's peak stall torque and reducer gear yield rating.
   - The **elbow joint** carries intermediate cantilever loads; $100\text{ Nm}$ represents maximum permissible bending moment.
   - The **wrist joint** maneuvers the end-effector with lower mechanical inertia; $40\text{ Nm}$ prevents shearing the harmonic drive flexspline.  
   Any instantaneous torque exceeding these limits indicates mechanical jamming or physical collision, requiring an immediate deterministic halt.

---

### 2. How the Model's Prediction and Maintenance Log Combine into a Final Decision

The system uses a hierarchical, priority-driven decision fusion engine that integrates two independent signals:
1. **Quantitative Telemetry Signal:** Processed through a Random Forest Classifier trained on multidimensional joint readings (torque, temperature, vibration) to predict mechanical wear (`NORMAL`, `MODERATE_WEAR`, `SEVERE_WEAR`).
2. **Qualitative Human Signal:** Unstructured technician inspection notes scanned via a keyword-based domain parser that categorizes observations into `CRITICAL_NOTE` (e.g., audible grinding, metal shavings, burning smell), `WARNING_NOTE` (e.g., clicking sounds, low grease, slight warmth), or `CLEAN_NOTE` (routine maintenance, cleaning, irrelevant notes).

The fusion logic combines these signals as follows:

- **EMERGENCY_STOP (Priority 1):** Triggered immediately if any raw sensor crosses a hard safety threshold. Overrides all other signals and models.
- **SCHEDULE_MAINTENANCE (Priority 2):** Triggered when high wear or damage is confirmed:
  - If the ML model predicts `SEVERE_WEAR`, maintenance is scheduled even if technicians haven't logged notes yet.
  - If technician notes report a `CRITICAL_NOTE` (e.g., metal shavings or grinding), maintenance is scheduled even if the sensor readings have not yet escalated to severe levels.
  - If the ML model predicts `MODERATE_WEAR` **and** technician notes independently record a `WARNING_NOTE`, the two corroborating signals escalate the status from observation to `SCHEDULE_MAINTENANCE`.
  - If a sensor spoofing attack or tampering is flagged, maintenance and security inspection are mandated.
- **MONITOR (Priority 3):** Triggered when an early warning exists on a single channel without confirmation from the other (e.g., ML predicts `MODERATE_WEAR` with a clean log, or technician noted minor clicking while sensors remain nominal). The arm continues production under heightened observation.
- **SAFE (Priority 4):** Assigned only when physical sensors are within nominal bounds, the ML model predicts `NORMAL` wear, and technician notes contain no adverse observations.

---

### 3. Spoofed-Sensor Defense: Mechanism, Capabilities, and Limitations

#### How the Defense Works
The defense leverages **thermodynamic and kinematic cross-sensor consistency**. When a joint suffers mechanical binding, gear wear, or bearing degradation, energy conservation dictates that mechanical work dissipated against friction produces heat ($\Delta T \propto \tau \cdot \omega$) and mechanical shock (vibration). 

If an attacker (or malicious network actor) alters sensor packets to inject an artificially low torque reading (e.g., claiming the shoulder draws only $10\text{ Nm}$ when normal is $50\text{--}80\text{ Nm}$ and distressed would be $>110\text{ Nm}$) in order to mask an overheating joint, the system performs a multi-variable physical correlation check:
$$\text{If } (T_{\text{joint}} \ge 67.0^\circ\text{C} \lor \text{Vib}_{\text{joint}} \ge 5.2\text{ mm/s}) \land (\tau_{\text{joint}} < \tau_{\text{min\_credible}}), \quad \implies \text{SPOOF\_ATTACK}$$

#### What Kind of Attack It Would Catch
1. **Torque Suppression / Masking Attacks:** An adversary intercepting CAN bus or telemetry packets and overwriting torque with low or zero values to trick predictive maintenance into reporting a "healthy" motor while the joint is physically grinding and overheating.
2. **Sensor Freeze / Replay Attacks:** Attacker replaying a previously recorded static "healthy" torque value while thermal and vibration sensors reflect current real-time stress.
3. **Sensor Disconnection / Drop-to-Zero Faults:** Accidental sensor wire detachment or open-circuit ADC faults where torque drops to $0\text{ Nm}$ during active high-load operation.

#### What Kind of Attack It Wouldn't Catch
1. **Coordinated Multi-Sensor Spoofing:** If an attacker has full root access to the sensor aggregator and simultaneously fakes torque, temperature, and vibration to all appear perfectly healthy and mutually consistent, this single-station cross-sensor defense cannot detect it without out-of-band validation (e.g., external optical cameras, main three-phase power meter telemetry, or cryptographic sensor attestation).
2. **Subtle In-Bounds Sensor Drift:** If an attacker introduces a minor bias ($\pm 3\%$) that remains within normal physical tolerances, it will not trigger the physical correlation threshold and must instead be caught by statistical residual analysis over long operational horizons.
