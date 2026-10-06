"""
=========================================================================================
 REINFORCEMENT LEARNING SELF-TUNING CONTROLLER: TRAINING PIPELINE
 Domain: Autonomous Vehicle Lateral Control & Proving Ground Adaptation
 Algorithm: Continuous Deep Actor-Critic Policy Gradient (A2C/PPO Style)
 Features: Stability-Preserving Safety Constraint Layer & Weight Export for MATLAB
=========================================================================================
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================================================
# 1. AUTONOMOUS VEHICLE 2-DOF DYNAMIC BICYCLE ENVIRONMENT
# =========================================================================================
class BicycleTrackingEnv:
    """
    Simulates a 4th-order 2-DOF dynamic bicycle model on varied road terrains.
    Observation Space: [e_y, d_e_y, delta_psi, d_e_psi, slope_feature, friction_feature] (dim=6)
    Action Space:      [Kp, Ki, Kd, Khead] (dim=4 adaptive gains)
    """
    def __init__(self, dt=0.01, episode_len_sec=20.0):
        self.dt = dt
        self.episode_len_sec = episode_len_sec
        self.max_steps = int(episode_len_sec / dt)
        
        # Vehicle Constants (matches MATLAB simulation)
        self.m = 1500.0       # Mass (kg)
        self.Iz = 3000.0      # Yaw inertia (kg*m^2)
        self.lf = 1.2         # CG to front axle (m)
        self.lr = 1.6         # CG to rear axle (m)
        self.L = self.lf + self.lr
        self.h_cg = 0.5       # CG height (m)
        self.Cf = 80000.0     # Front cornering stiffness (N/rad)
        self.Cr = 80000.0     # Rear cornering stiffness (N/rad)
        self.Vx = 10.0        # Longitudinal speed (m/s) [36 km/h]
        self.g = 9.81         # Gravity (m/s^2)
        self.delta_max = 0.5  # Max steer angle (rad)
        self.d_delta_max = 0.6 # Max steer slew rate (rad/s)
        
    def reset(self, randomize_env=True):
        self.step_idx = 0
        self.t = 0.0
        
        # Randomized or nominal environmental parameters per episode
        if randomize_env:
            self.slope_deg = np.random.uniform(-10.0, 10.0)
            self.mu_road = np.random.uniform(0.25, 0.85)
            self.curve_freq = np.random.uniform(0.15, 0.35)
            self.curve_amp = np.random.uniform(0.03, 0.08)
        else:
            self.slope_deg = 0.0
            self.mu_road = 0.85
            self.curve_freq = 0.20
            self.curve_amp = 0.05
            
        # Vehicle States
        self.X = 0.0
        self.Y = 0.0
        self.psi = 0.0
        self.Vy = 0.0
        self.r = 0.0
        
        # Reference states
        self.X_ref = 0.0
        self.Y_ref = 0.0
        self.psi_ref = 0.0
        self.r_ref = 0.0
        
        # Controller internal states
        self.int_ey = 0.0
        self.prev_delta = 0.0
        
        return self._get_obs()

    def _get_obs(self):
        # Tracking errors
        ex_g = self.X_ref - self.X
        ey_g = self.Y_ref - self.Y
        self.ey = ey_g * np.cos(self.psi) - ex_g * np.sin(self.psi)
        
        dpsi_raw = self.psi_ref - self.psi
        self.delta_psi = np.arctan2(np.sin(dpsi_raw), np.cos(dpsi_raw))
        
        self.d_ey = -self.Vy * np.cos(self.psi) - self.Vx * np.sin(self.psi)
        self.d_e_psi = self.r - self.r_ref
        
        # Normalized environmental features
        slope_feat = self.slope_deg / 10.0
        fric_feat = (self.mu_road - 0.55) / 0.35
        
        obs = np.array([self.ey, self.d_ey, self.delta_psi, self.d_e_psi, slope_feat, fric_feat], dtype=np.float32)
        return obs

    def step(self, raw_action, apply_safety_constraint=True):
        self.step_idx += 1
        self.t += self.dt
        
        # --- A. STABILITY-PRESERVING SAFETY CONSTRAINT LAYER ---
        if apply_safety_constraint:
            # 1. Bounded Action Space Projection (Option A)
            Kp = np.clip(raw_action[0], 0.4, 2.2)
            Ki = np.clip(raw_action[1], 0.01, 0.15)
            Kd = np.clip(raw_action[2], 0.02, 0.45)
            Khead = np.clip(raw_action[3], 0.6, 1.8)
            
            # 2. Dynamic Low-Friction Damping Injection
            if self.mu_road < 0.6:
                Kd = Kd * (1.0 + 1.2 * (0.6 - self.mu_road))
                Ki = Ki * (self.mu_road / 0.85)
                
            # 3. Dynamic Slope Incline Compensation
            if self.slope_deg > 2.0:
                Kp = Kp * (1.0 + 0.025 * self.slope_deg)
                Khead = Khead * (1.0 + 0.02 * self.slope_deg)
                
            # 4. Anti-Windup Error Clamping
            self.int_ey = np.clip(self.int_ey + self.ey * self.dt, -1.5, 1.5)
        else:
            # Unconstrained Baseline (raw unbounded actor outputs)
            Kp, Ki, Kd, Khead = raw_action[0], raw_action[1], raw_action[2], raw_action[3]
            self.int_ey += self.ey * self.dt

        # --- B. COMPUTE STEERING CONTROL SIGNAL ---
        u_pid = Kp * self.ey + Ki * self.int_ey + Kd * self.d_ey
        u_total = u_pid + Khead * self.delta_psi
        
        # Curvature Feedforward
        kappa = self.r_ref / self.Vx
        u_total += self.L * kappa
        
        # Actuator Slew Rate Limiter & Saturation
        max_rate = self.d_delta_max * self.dt
        delta_cmd = self.prev_delta + np.clip(u_total - self.prev_delta, -max_rate, max_rate)
        delta = np.clip(delta_cmd, -self.delta_max, self.delta_max)
        self.prev_delta = delta
        
        # --- C. 2-DOF DYNAMIC BICYCLE INTEGRATION ---
        # Axle normal loads under terrain slope
        th_rad = np.radians(self.slope_deg)
        Fzf = (self.m * self.g * (self.lr * np.cos(th_rad) - self.h_cg * np.sin(th_rad))) / self.L
        Fzr = (self.m * self.g * (self.lf * np.cos(th_rad) + self.h_cg * np.sin(th_rad))) / self.L
        Fyf_max = max(self.mu_road * Fzf, 100.0)
        Fyr_max = max(self.mu_road * Fzr, 100.0)
        
        # Tire Slip Angles
        alpha_f = delta - np.arctan2(self.Vy + self.lf * self.r, self.Vx)
        alpha_r = - np.arctan2(self.Vy - self.lr * self.r, self.Vx)
        
        # Lateral Tire Forces
        Fyf = np.clip(self.Cf * alpha_f, -Fyf_max, Fyf_max)
        Fyr = np.clip(self.Cr * alpha_r, -Fyr_max, Fyr_max)
        
        # Equations of Motion
        ay = (Fyf * np.cos(delta) + Fyr) / self.m
        d_Vy = ay - self.Vx * self.r
        d_r = (self.lf * Fyf * np.cos(delta) - self.lr * Fyr) / self.Iz
        d_psi = self.r
        d_X = self.Vx * np.cos(self.psi) - self.Vy * np.sin(self.psi)
        d_Y = self.Vx * np.sin(self.psi) + self.Vy * np.cos(self.psi)
        
        self.Vy += d_Vy * self.dt
        self.r += d_r * self.dt
        self.psi += d_psi * self.dt
        self.X += d_X * self.dt
        self.Y += d_Y * self.dt
        
        # Update Reference Trajectory
        delta_ref = self.curve_amp * np.sin(self.curve_freq * self.t)
        self.r_ref = (self.Vx / self.L) * np.tan(delta_ref)
        self.psi_ref += self.r_ref * self.dt
        self.X_ref += self.Vx * np.cos(self.psi_ref) * self.dt
        self.Y_ref += self.Vx * np.sin(self.psi_ref) * self.dt
        
        next_obs = self._get_obs()
        
        # --- D. REWARD & STABILITY TERMINATION ---
        # Tracking precision, smooth steering, low yaw deviation
        reward = -(15.0 * (self.ey ** 2) + 2.0 * (self.d_ey ** 2) + 8.0 * (self.delta_psi ** 2) + 1.0 * (delta ** 2))
        
        # Instability detection (spin-out / large divergence)
        done = False
        is_instability = False
        if abs(self.ey) > 1.5 or abs(alpha_f) > np.radians(15.0) or abs(alpha_r) > np.radians(15.0):
            done = True
            is_instability = True
            reward -= 250.0 # Heavy crash/instability penalty
            
        if self.step_idx >= self.max_steps:
            done = True
            
        return next_obs, reward, done, is_instability, [Kp, Ki, Kd, Khead]

# =========================================================================================
# 2. ACTOR AND CRITIC NEURAL NETWORKS
# =========================================================================================
class ActorNetwork(nn.Module):
    """
    Policy Network: Maps 6-dim state observation to 4 adaptive gains.
    Matches the exact 6x8x4 architecture in rl_adaptive_controller.m.
    """
    def __init__(self, obs_dim=6, action_dim=4):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 8)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(8, action_dim)
        
        # Initialize output bias around nominal baseline gains [0.80, 0.05, 0.08, 1.00]
        self.fc2.bias.data = torch.tensor([0.80, 0.05, 0.08, 1.00], dtype=torch.float32)
        
    def forward(self, x):
        h = self.tanh(self.fc1(x))
        out = self.fc2(h)
        return out

class CriticNetwork(nn.Module):
    """
    Value Network: Evaluates expected state return for advantage estimation.
    """
    def __init__(self, obs_dim=6):
        super(CriticNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        return self.net(x)

# =========================================================================================
# 3. TRAINING PIPELINE WITH SAFETY CONSTRAINT ABLATION
# =========================================================================================
def train_rl_controller(num_episodes=150, apply_safety=True):
    env = BicycleTrackingEnv(dt=0.01, episode_len_sec=15.0)
    actor = ActorNetwork().to(device)
    critic = CriticNetwork().to(device)
    
    actor_opt = optim.Adam(actor.parameters(), lr=1e-3)
    critic_opt = optim.Adam(critic.parameters(), lr=3e-3)
    
    episode_rewards = []
    total_violations = 0
    violation_history = []
    
    print(f"\n>>> Starting RL Training ({'CONSTRAINED (With Safety Layer)' if apply_safety else 'UNCONSTRAINED BASELINE'})...")
    print("-" * 80)
    
    for ep in range(1, num_episodes + 1):
        obs = env.reset(randomize_env=True)
        ep_reward = 0.0
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        done = False
        while not done:
            state_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            raw_action = actor(state_t).squeeze(0)
            
            # Add decaying exploration noise
            noise = torch.randn_like(raw_action) * max(0.05, 0.30 * (1.0 - ep / num_episodes))
            action_exec = (raw_action + noise).detach().cpu().numpy()
            
            next_obs, reward, done, is_instability, applied_gains = env.step(action_exec, apply_safety_constraint=apply_safety)
            
            if is_instability:
                total_violations += 1
                
            states.append(obs)
            actions.append(action_exec)
            rewards.append(reward)
            next_states.append(next_obs)
            dones.append(done)
            
            obs = next_obs
            ep_reward += reward
            
        # Update Actor-Critic via Advantage Policy Gradient
        states_t = torch.FloatTensor(np.array(states)).to(device)
        rewards_t = torch.FloatTensor(np.array(rewards)).to(device)
        dones_t = torch.FloatTensor(np.array(dones)).to(device)
        
        values = critic(states_t).squeeze(-1)
        
        # Compute discounted returns
        returns = []
        R = 0.0
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + 0.99 * R * (1.0 - float(d))
            returns.insert(0, R)
        returns_t = torch.FloatTensor(returns).to(device)
        
        # Advantage
        advantage = returns_t - values.detach()
        
        # Critic Loss
        critic_loss = nn.MSELoss()(values, returns_t)
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()
        
        # Actor Loss
        pred_actions = actor(states_t)
        target_actions = torch.FloatTensor(np.array(actions)).to(device)
        action_loss = nn.MSELoss()(pred_actions, target_actions)
        actor_loss = action_loss * advantage.mean()
        
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()
        
        episode_rewards.append(ep_reward)
        violation_history.append(total_violations)
        
        if ep % 25 == 0 or ep == 1:
            print(f"Episode {ep:3d}/{num_episodes} | Return: {ep_reward:8.1f} | Total Stability Violations: {total_violations}")
            
    print("-" * 80)
    print(f"Training Complete! Final Stability Violations: {total_violations}\n")
    return actor, episode_rewards, violation_history

# =========================================================================================
# 4. EXPORT WEIGHTS FOR MATLAB (`exported_rl_weights.m`)
# =========================================================================================
def export_weights_to_matlab(actor, filepath="D:/Projects/CS_UPDATED/exported_rl_weights.m"):
    """
    Saves the learned Actor network weights in pure MATLAB syntax.
    """
    W1 = actor.fc1.weight.detach().cpu().numpy()
    b1 = actor.fc1.bias.detach().cpu().numpy().reshape(-1, 1)
    W2 = actor.fc2.weight.detach().cpu().numpy()
    b2 = actor.fc2.bias.detach().cpu().numpy().reshape(-1, 1)
    
    with open(filepath, "w") as f:
        f.write("% =========================================================================\n")
        f.write("% AUTO-GENERATED RL ACTOR POLICY WEIGHTS (EXPORTED FROM train_rl_agent.py)\n")
        f.write("% =========================================================================\n\n")
        
        f.write("% Layer 1: Weights W1 (8 x 6) & Bias b1 (8 x 1)\n")
        f.write("W1 = [\n")
        for row in W1:
            f.write("    " + ", ".join([f"{val:8.4f}" for val in row]) + ";\n")
        f.write("];\n\n")
        
        f.write("b1 = [" + "; ".join([f"{val[0]:.4f}" for val in b1]) + "];\n\n")
        
        f.write("% Layer 2: Weights W2 (4 x 8) & Bias b2 (4 x 1)\n")
        f.write("W2 = [\n")
        for row in W2:
            f.write("    " + ", ".join([f"{val:8.4f}" for val in row]) + ";\n")
        f.write("];\n\n")
        
        f.write("b2 = [" + "; ".join([f"{val[0]:.4f}" for val in b2]) + "];\n")
        
    print(f"[SUCCESS] Exported MATLAB-ready weights to: {filepath}")

# =========================================================================================
# 5. MAIN EXECUTION
# =========================================================================================
if __name__ == "__main__":
    t_start = time.time()
    
    # 1. Train Constrained Agent
    trained_actor, rewards, violations = train_rl_controller(num_episodes=150, apply_safety=True)
    
    # 2. Export Model Checkpoint & MATLAB Weights
    torch.save(trained_actor.state_dict(), "D:/Projects/CS_UPDATED/rl_actor_checkpoint.pth")
    print("[SUCCESS] Saved PyTorch Checkpoint to: D:/Projects/CS_UPDATED/rl_actor_checkpoint.pth")
    
    export_weights_to_matlab(trained_actor, "D:/Projects/CS_UPDATED/exported_rl_weights.m")
    
    print(f"\nEntire RL Training Pipeline Finished in {time.time() - t_start:.2f} seconds!")
