from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from egga.rl.env import EGGAEnv
from egga.rl.model import ActorCritic


@dataclass
class PPOConfig:
    total_timesteps: int = 15000
    learning_rate: float = 3e-4
    n_steps: int = 1000
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


class PPOTrainer:
    """PPO implementation with GAE and per-sample clipping for EGGA."""

    def __init__(self, env: EGGAEnv, model: ActorCritic, cfg: PPOConfig | None = None) -> None:
        self.env = env
        self.model = model
        self.cfg = cfg or PPOConfig()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate, eps=1e-5)
        self.rng = np.random.default_rng()

    def train(self, log_callback: Any = None) -> list[dict[str, float]]:
        cfg = self.cfg
        obs_dim = self.model.obs_dim
        act_dim = self.model.act_dim

        obs_buf = np.zeros((cfg.n_steps, obs_dim), dtype=np.float32)
        act_buf = np.zeros((cfg.n_steps, act_dim), dtype=np.float32)
        logp_buf = np.zeros(cfg.n_steps, dtype=np.float32)
        rew_buf = np.zeros(cfg.n_steps, dtype=np.float32)
        val_buf = np.zeros(cfg.n_steps, dtype=np.float32)
        done_buf = np.zeros(cfg.n_steps, dtype=np.float32)

        metrics_history: list[dict[str, float]] = []
        global_step = 0
        obs, _ = self.env.reset()

        while global_step < cfg.total_timesteps:
            ep_rewards: list[float] = []
            ep_len = 0
            cur_ep_rew = 0.0

            # 1. Rollout collection
            for step in range(cfg.n_steps):
                global_step += 1
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action, log_prob, value = self.model.get_action(obs_tensor)

                act_np = action.squeeze(0).cpu().numpy()
                logp_np = log_prob.squeeze(0).cpu().numpy()
                val_np = value.squeeze(0).cpu().numpy()

                next_obs, reward, terminated, truncated, _ = self.env.step(act_np)
                done = terminated or truncated

                obs_buf[step] = obs
                act_buf[step] = act_np
                logp_buf[step] = logp_np
                rew_buf[step] = reward
                val_buf[step] = val_np
                done_buf[step] = float(done)

                obs = next_obs
                cur_ep_rew += reward
                ep_len += 1

                if done:
                    ep_rewards.append(cur_ep_rew)
                    cur_ep_rew = 0.0
                    obs, _ = self.env.reset()

            # 2. GAE computation
            with torch.no_grad():
                last_obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                last_val = self.model.get_value(last_obs_tensor).item()

            advantages = np.zeros(cfg.n_steps, dtype=np.float32)
            last_gae = 0.0
            for t in reversed(range(cfg.n_steps)):
                if t == cfg.n_steps - 1:
                    next_non_terminal = 1.0 - done_buf[t]
                    next_value = last_val
                else:
                    next_non_terminal = 1.0 - done_buf[t]
                    next_value = val_buf[t + 1]
                delta = rew_buf[t] + cfg.gamma * next_value * next_non_terminal - val_buf[t]
                last_gae = delta + cfg.gamma * cfg.gae_lambda * next_non_terminal * last_gae
                advantages[t] = last_gae
            returns = advantages + val_buf

            # Normalize advantages
            adv_mean = np.mean(advantages)
            adv_std = np.std(advantages) + 1e-8
            norm_advantages = (advantages - adv_mean) / adv_std

            # Convert to PyTorch tensors
            b_obs = torch.tensor(obs_buf, dtype=torch.float32)
            b_act = torch.tensor(act_buf, dtype=torch.float32)
            b_logp = torch.tensor(logp_buf, dtype=torch.float32)
            b_adv = torch.tensor(norm_advantages, dtype=torch.float32)
            b_ret = torch.tensor(returns, dtype=torch.float32)

            # 3. PPO Epochs
            clip_fracs = []
            approx_kls = []
            actor_losses = []
            critic_losses = []

            n_samples = cfg.n_steps
            indices = np.arange(n_samples)

            for _epoch in range(cfg.n_epochs):
                self.rng.shuffle(indices)
                for start in range(0, n_samples, cfg.batch_size):
                    end = start + cfg.batch_size
                    mb_idx = indices[start:end]

                    new_logp, entropy, new_val = self.model.evaluate_actions(
                        b_obs[mb_idx], b_act[mb_idx]
                    )

                    log_ratio = new_logp - b_logp[mb_idx]
                    ratio = torch.exp(log_ratio)

                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                        approx_kls.append(approx_kl)
                        clip_frac = ((ratio - 1.0).abs() > cfg.clip_range).float().mean().item()
                        clip_fracs.append(clip_frac)

                    # Clipped surrogate actor loss
                    mb_adv = b_adv[mb_idx]
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * mb_adv
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value function loss
                    val_loss = 0.5 * ((new_val - b_ret[mb_idx]) ** 2).mean()

                    # Total loss
                    entropy_loss = -entropy.mean()
                    total_loss = policy_loss + cfg.vf_coef * val_loss + cfg.ent_coef * entropy_loss

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                    self.optimizer.step()

                    actor_losses.append(policy_loss.item())
                    critic_losses.append(val_loss.item())

            iter_metrics = {
                "step": float(global_step),
                "mean_reward": float(np.mean(ep_rewards))
                if ep_rewards
                else float(np.mean(rew_buf)),
                "policy_loss": float(np.mean(actor_losses)),
                "value_loss": float(np.mean(critic_losses)),
                "approx_kl": float(np.mean(approx_kls)),
                "clip_fraction": float(np.mean(clip_fracs)),
            }
            metrics_history.append(iter_metrics)
            if log_callback:
                log_callback(iter_metrics)

        return metrics_history
