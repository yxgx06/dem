# Model Card: B5 Envelope-Guarded Gain Adaptation Policy

**Model Identifier**: `egga-b5-ppo-actor-v1`  
**Model Type**: Multi-Layer Perceptron (MLP) for Gain Perturbation Proposal  
**Date**: October 2026  
**License**: Research Prototype / Automotive Safety Benchmark  

---

## 1. Model Overview

The **B5 Actor** is a compact 92-parameter neural network trained via Proximal Policy Optimization (PPO) to propose fine-grained feedback gain adjustments $\Delta K = [\Delta K_p, \Delta K_i, \Delta K_d, \Delta K_{\text{head}}]$ for an automated vehicle steering controller.

The model is designed from inception under **safety-by-construction** principles: it does **not** output steering actuator angles directly. Instead, its proposed parameter modifications are filtered, checked, and projected by the verified runtime supervisor into a precomputed Lyapunov stability envelope.

```
+-------------------------------------------------------------------------------+
|                             B5 CLOSED-LOOP ARCHITECTURE                       |
|                                                                               |
|   +-----------------------+              +--------------------------------+   |
|   | Estimators & Sensors  |              | B5 Actor MLP (92 params)       |   |
|   |  - [e_y, e_dot]       |              |  - 6 inputs                    |   |
|   |  - [psi_err, r_err]   | -----------> |  - 8 hidden (tanh)             |   |
|   |  - [mu_lo, tau_bar]   |              |  - 4 outputs (tanh)            |   |
|   +-----------------------+              +--------------------------------+   |
|              |                                           |                    |
|              |                                           | Proposed Delta K   |
|              v                                           v                    |
|   +-----------------------------------------------------------------------+   |
|   |                   VERIFIED RUNTIME SUPERVISOR                         |   |
|   |  - Check slew limits: |Delta K_t - Delta K_{t-1}| <= rate_max         |   |
|   |  - Cell mask lookup: Verified gain table [800 x 19 uint64]            |   |
|   |  - Nearest-verified projection: K_applied in Envelope                 |   |
|   |  - Lateral acceleration speed cap & steering limits                   |   |
|   +-----------------------------------------------------------------------+   |
|                                      |                                        |
|                                      v K_applied, delta_max, v_cmd            |
|                            [ Vehicle Plant / ECU ]                            |
+-------------------------------------------------------------------------------+
```

---

## 2. Model Architecture & Specifications

| Attribute | Specification | Details |
|---|---|---|
| **Input Dimensions** | 6 continuous signals | $e_y$ (m), $\dot{e}_y$ (m/s), $\tilde{\psi}$ (rad), $\tilde{r}$ (rad/s), $\hat{\mu}_{\text{lo}}$, $\bar{\tau}$ (s) |
| **Observation Isolation** | Strict | Ground truth plant variables (slip angles, true friction) are strictly sequestered |
| **Hidden Layer** | 1 layer, 8 neurons | Fully connected, $\tanh$ activation |
| **Output Layer** | 4 neurons | Fully connected, $\tanh$ activation |
| **Action Bounds** | $[-1.0, 1.0]^4$ | Scaled to physical gain bounds: $\pm [0.60, 0.04, 0.15, 0.60]$ |
| **Parameter Count** | **92 float32 parameters** | Weights: $6 \times 8 + 8 \times 4 = 80$; Biases: $8 + 4 = 12$ |
| **ROM Footprint** | **368 Bytes** | Stored in `.rodata` (`src/c/actor_weights.h`) |
| **Execution Latency** | **< 1.5 μs / tick** | Host latency: 1.46 μs mean (combined Actor + Supervisor) |
| **Memory Allocation** | **0 Bytes Heap** | Pure stack evaluation; zero `malloc` / `free` |

---

## 3. Training & Data Provenance

- **Training Algorithm**: PPO (Proximal Policy Optimization) with Generalized Advantage Estimation (GAE).
- **Training Campaign**: 20 independently trained random seeds (`seeds: 2026..2045`).
- **Training Distribution**: 48 slice-disjoint scenarios (`experiments/scenario_sets/train.json`) constructed with Latin-hypercube parameter slicing.
- **Domain Randomization**:
  - Friction coefficient $\mu \in [0.25, 0.95]$
  - Steering actuator latency $\tau \in [0.01, 0.09]\text{ s}$
  - Vehicle payload mass $m \in [1.0, 1.6] \times m_{\text{nom}}$
  - Path curvature $\kappa \in [-0.04, 0.04]\text{ m}^{-1}$
- **Reward Function**: Quadratic penalty on lateral tracking error $e_y^2$, heading error $\tilde{\psi}^2$, and gain change rate $(\Delta K_t - \Delta K_{t-1})^2$, with heavy penalty on supervisor safety intervention.

---

## 4. Evaluation & Safety Benchmarks

The B5 model was evaluated across 120 total scenarios spanning three slice-disjoint datasets:

| Dataset | Scenarios | B5 Divergences | B3 (Ablation) Divergences | B5 p95 Tracking Error | Status |
|---|---|---|---|---|---|
| **Train Set** | 48 | **0 / 48 (0.0%)** | 3 / 48 (6.2%) | 3.06 cm | Certified |
| **Validation Set** | 24 | **0 / 24 (0.0%)** | 4 / 24 (16.7%) | 3.93 cm | Certified |
| **Held-Out Test Set** | 48 | **0 / 48 (0.0%)** | **31 / 48 (64.6%)** | **1.45 cm** | Pre-registered Pass |

### Key Safety Finding: Supervisor Necessity
In the B3 ablation (identical RL policy without supervisor envelope guarding), the system experienced **31 catastrophic divergences** on the held-out test set under actuator latency and low road friction. The B5 architecture, incorporating runtime envelope guarding, completely eliminated all 31 failures, achieving **zero divergences**.

---

## 5. Intended Use & Deployment Constraints

### Intended Use
- Closed-loop lateral path tracking for highway and rural automated driving ($v \le 15\text{ m/s}$).
- Deployment within the verified EGGA supervisor C99 runtime engine (`src/c/`).
- Real-time embedded ECUs conforming to ISO 26262 ASIL D requirements.

### Out-of-Scope / Prohibited Uses
- **Standalone execution**: The B5 policy must never command steering actuators directly without supervisor envelope projection.
- **Unverified speed regimes**: Operating at speeds $v > 15.0\text{ m/s}$ (outside precomputed envelope grid).
- **Extreme friction floor**: Operating on surfaces with $\mu < 0.20$ (system must enter `MINIMAL_RISK` safe stop).
- **Off-road / reverse driving**: Vehicle dynamics outside standard planar bicycle model domain.
