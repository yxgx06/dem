# Hazard Analysis and Risk Assessment (HARA)
**Document Ref**: EGGA-SAF-HARA-001  
**Standard**: ISO 26262-3:2018 (Road vehicles — Functional safety — Part 3: Concept phase)  
**Item Definition**: Envelope-Guarded Gain Adaptation (EGGA) Lateral Control System  

---

## 1. Item Definition & System Boundary

The **EGGA (Envelope-Guarded Gain Adaptation)** lateral control subsystem is responsible for steering control augmentation in autonomous and automated driving applications. The item operates at 100 Hz ($dt = 0.01\text{ s}$) within the vehicle electronic control unit (ECU) architecture:

```
[Vision / Path Planning] ---> (Curvature kappa, Speed Req) ---+
[Tire & Inertia EKF]    ---> (mu_lo, mu_hi, mass_hi)         |
[Delay Estimator]       ---> (tau_bar)                      v
[Vehicle CAN Bus]       ---> (v, r, delta_meas, e_y, e_dot) -> [ EGGA SUPERVISOR ] ---> (delta_cmd, v_cmd) -> [Steer Actuator]
[PPO Actor (92 params)] ---> (Proposed Delta K)              -+
```

### System Interfaces & Operational Limits
- **Inputs**: Measured vehicle speed $v$, lateral error $e_y$, heading error $\tilde{\psi}$, yaw rate $r$, steering angle $\delta_{\text{meas}}$, road curvature $\kappa$, estimated friction interval $[\mu_{\text{lo}}, \mu_{\text{hi}}]$, estimated delay bound $\bar{\tau}$, estimated mass upper bound $\hat{m}_{\text{hi}}$, proposed RL gain adjustment $\Delta K$.
- **Outputs**: Verified feedback gains $K = [K_p, K_i, K_d, K_{\text{head}}]$, friction-consistent speed limit $v_{\text{cmd}}$, steering angle limit $\delta_{\max}$, steering slew rate limit $\dot{\delta}_{\max}$, operational mode, and diagnostic flags.
- **Fault-Tolerant Time Interval (FTTI)**: Defined as $100\text{ ms}$ (10 control cycles) at high speeds before an uncommanded lateral disturbance forces lane departure.

---

## 2. Hazard Identification & Operational Situations

We analyze potential hazards caused by malfunctioning behavior or functional insufficiencies of the steering control and gain adaptation system:

| Hazard ID | Potential Malfunction / Insufficiency | Operational Situation | Hazardous Event Description |
|---|---|---|---|
| **HZ-01** | Unintended excessive steering jerk / high feedback gain | High-speed highway driving ($v = 15\text{ m/s}$, dry asphalt) | Closed-loop parameter instability causing sudden severe lateral excursion into adjacent lane or barrier. |
| **HZ-02** | Command exceeds available tire adhesion | Wet/icy curve ($v = 10\text{ m/s}, \mu < 0.30, \kappa = 0.02\text{ m}^{-1}$) | Command lateral acceleration exceeds friction limit ($v^2 \kappa > \mu g$), resulting in front axle saturation, plow understeer, and off-road departure. |
| **HZ-03** | Delay-induced closed-loop limit cycling | Steering actuator lag or CAN jitter ($\tau > 80\text{ ms}$) | Phase lag converts negative feedback into positive feedback, causing violent lateral weave and vehicle spinout. |
| **HZ-04** | Operation on corrupted or stale state estimates | Transition from straight to curved road after straight cruising | Controller relies on stale friction estimate; fails to cap speed ahead of hairpin curve. |

---

## 3. Classification: Severity, Exposure, Controllability, and ASIL

Per ISO 26262-3 Table 4:

| Hazard ID | Situation Description | Severity (S) | Exposure (E) | Controllability (C) | ASIL Rating |
|---|---|---|---|---|---|
| **HZ-01** | High-speed uncommanded steering jerk | **S3** (Fatal injuries from high-speed rollover / head-on) | **E4** (Standard operational driving mode) | **C3** (Driver reaction time > 200 ms; uncontrollable) | **ASIL D** |
| **HZ-02** | Tire force saturation on slippery curve | **S3** (Severe roadway departure / collision with obstacles) | **E3** (Wet/winter driving conditions common) | **C3** (Physical loss of tire adhesion; steering ineffective) | **ASIL D** |
| **HZ-03** | Limit-cycling lateral weave under latency | **S2** (Severe injury from high-speed loss of lane control) | **E3** (Component aging, network congestion) | **C2** (Experienced driver may brake to recover) | **ASIL C** |
| **HZ-04** | Late braking / over-speed on sharp curve | **S3** (High-energy off-road trajectory) | **E4** (Highway-to-ramp transitions) | **C3** (Vehicle dynamics exceed physical tire envelope) | **ASIL D** |

---

## 4. Safety Goals & Derived Technical Safety Requirements

From the identified ASIL D and ASIL C hazards, we formulate the top-level safety goals:

### Safety Goal 1 (SG-01) — ASIL D
> **Prevent uncommanded steering deviations or closed-loop lateral divergence ($|e_y| > 1.5\text{ m}$) resulting from adaptive gain selection under all in-domain operating conditions.**
- **TSR-01.1**: The runtime supervisor shall guarantee that applied controller gains $K$ belong strictly to the offline Lyapunov-certified stability envelope table for the current estimated operating slice $(\hat{v}, \hat{\mu}, \bar{\tau}, \hat{m})$.
- **TSR-01.2**: Changes in applied feedback gains between consecutive control ticks shall be constrained to at most one adjacent grid cell hop every `gain_hop_period` ($0.1\text{ s}$).
- **TSR-01.3**: Proposed RL gain adjustments $\Delta K$ must undergo range checks and slew rate limit verification prior to projection; unverified proposals shall be discarded.

### Safety Goal 2 (SG-02) — ASIL D
> **Prevent loss of lateral adhesion by restricting commanded vehicle speed and steering demands to physical tire-road friction capabilities.**
- **TSR-02.1**: Commanded vehicle speed $v_{\text{cmd}}$ shall not exceed the friction speed ceiling $v_{\text{cap}} = \sqrt{a_{y,\max} / \kappa}$ with $a_{y,\max} = k \cdot \mu_{\text{lo}} \cdot g$.
- **TSR-02.2**: Commanded steering angle $\delta_{\text{cmd}}$ shall be clamped to the understeer-compensated tire saturation angle $\delta_{\max}(v, \mu_{\text{eff}}, m)$.
- **TSR-02.3**: Steering slew rate $\dot{\delta}$ shall not exceed envelope-certified safe rate limits.

### Safety Goal 3 (SG-03) — ASIL C
> **Maintain closed-loop lateral stability in the presence of actuator delay, degradation, or communication latency.**
- **TSR-03.1**: Actuator tracking error $|\delta_{\text{cmd}}(t - \tau) - \delta_{\text{meas}}(t)|$ shall be monitored continuously. Persistent residuals exceeding threshold shall trigger the `FLAG_ACTUATOR` fault and escalate mode to `FALLBACK`.
- **TSR-03.2**: When estimated actuator lag $\bar{\tau} > \tau_{\text{enter}}$, the system shall transition to `DEGRADED_ACTUATOR` mode and restrict gain selection to delay-tolerant envelope subsets.

### Safety Goal 4 (SG-04) — ASIL D
> **Ensure safe deterministic fallback and minimal-risk safe stop upon state estimator failure or operational domain violation.**
- **TSR-04.1**: Estimator drop-out, stale CAN inputs, or persistent bicycle-model yaw rate discrepancies shall escalate mode to `FALLBACK` within 1 control cycle ($10\text{ ms}$).
- **TSR-04.2**: Encountering operational states below the friction floor ($\mu < \mu_{\text{floor}} = 0.20$) or with an empty verified gain set shall immediately trigger `MINIMAL_RISK` safe stopping.
- **TSR-04.3**: `MINIMAL_RISK` mode shall latch and only clear when continuous fault-free operation exceeds `stale_grace` ($0.3\text{ s}$) and an explicit reset command is asserted.

---

## 5. Verification & Traceability Summary

All 4 Safety Goals and derived Technical Safety Requirements are formally mapped in [traceability.csv](file:///D:/Projects/CS_UPDATED/docs/safety/traceability.csv) and validated through the 252-test automated verification suite, achieving 100.00% branch coverage on the supervisor core.
