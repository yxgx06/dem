# Reinforcement-Learning-Based Self-Tuning PID/MPC Controller

An autonomous vehicle control framework combining **Reinforcement Learning (RL)**, **Model Predictive Control (MPC)**, and **Self-Tuning PID** with a stability-preserving safety constraint layer for vehicle path tracking across complex, multi-terrain proving ground scenarios.

---

## 🚀 Key Highlights & Features

- **Multi-Terrain Mission Simulation**: Evaluates vehicle dynamics across 5 continuous proving ground stages:
  1. *Flat Road, Dry Asphalt* ($\theta=0^\circ, \mu=0.85$) - High-speed cruising
  2. *Steep Uphill* ($\theta=+10^\circ, \mu=0.85$) - Rear load transfer
  3. *Uphill with Rain* ($\theta=+10^\circ, \mu=0.50$) - Reduced adhesion S-curves
  4. *Downhill on Black Ice* ($\theta=-10^\circ, \mu=0.25$) - High-slip downhill tracking
  5. *Emergency Double Lane Change* ($\mu=0.60$) - Dynamic obstacle avoidance
- **Safety-Constrained RL Layer**: Eliminates instability and state-boundary violations during reinforcement learning policy optimization.
- **Multi-Controller Comparison**: Benchmarks Fixed PID vs. Model Predictive Control (MPC) vs. RL Adaptive Controller.
- **Simulink Integration**: Native MATLAB/Simulink 2-DOF dynamic bicycle model integration.

---

## 📁 Repository Structure

```
.
├── BicyclePathTracking1.slx                    # Simulink 2-DOF vehicle path tracking model
├── exported_rl_weights.m                       # Exported trained RL actor network weights
├── Figure_Lateral_Tracking_Benchmark.jpeg      # Multi-terrain tracking error benchmark plot
├── Figure_Step_Response_Benchmark.png          # Step response and settling performance plot
├── Figure_Training_Safety_Ablation.png         # Safety layer ablation & learning convergence plot
├── generate_missing_benchmarks.py             # Python script for step response & ablation benchmarking
├── master_bicycle_simulation.m                # Master bicycle simulation runner in MATLAB
├── mission_proving_ground_rl.m                # 75-second automotive proving ground mission test
├── rl_actor_checkpoint.pth                    # PyTorch RL actor neural network weights
├── rl_adaptive_controller.m                   # MATLAB adaptive controller implementation
├── RL_PID_Controller_Project.docx             # Project proposal & technical report document
├── run_in_simulink.m                          # Simulink automation execution script
└── train_rl_agent.py                          # PyTorch RL training script with safety constraints
```

---

## 🛠️ Usage Instructions

### 1. MATLAB & Simulink
1. Open MATLAB and navigate to the project directory.
2. Run the multi-terrain proving ground benchmark:
   ```matlab
   mission_proving_ground_rl
   ```
3. To run the dynamic bicycle model simulation:
   ```matlab
   master_bicycle_simulation
   ```
4. To execute the Simulink model directly:
   ```matlab
   run_in_simulink
   ```

### 2. Python Reinforcement Learning & Benchmarking
1. Install dependencies:
   ```bash
   pip install torch numpy matplotlib
   ```
2. Train the RL agent:
   ```bash
   python train_rl_agent.py
   ```
3. Generate benchmark plots:
   ```bash
   python generate_missing_benchmarks.py
   ```
