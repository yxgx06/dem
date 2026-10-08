from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class ActorCritic(nn.Module):
    """6 -> 8 (tanh) -> 4 MLP scheduling [Kp, Ki, Kd, Khead] with 92 parameters.

    Actor architecture:
      Linear(6, 8): 6 * 8 + 8 = 56 params
      Tanh()
      Linear(8, 4): 8 * 4 + 4 = 36 params
      Total actor parameters = 92 parameters.
    """

    def __init__(self, obs_dim: int = 6, act_dim: int = 4) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # 92-parameter Actor MLP
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 8),
            nn.Tanh(),
            nn.Linear(8, act_dim),
        )
        # Trainable log-std for Gaussian policy exploration
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # Critic value network (used during training only)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.actor.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Small gain on output layer to start near zero delta-K
        nn.init.orthogonal_(self.actor[2].weight, gain=0.01)

        for m in self.critic.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.critic[4].weight, gain=1.0)

    def count_actor_params(self) -> int:
        return sum(p.numel() for p in self.actor.parameters())

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        if deterministic:
            action = torch.tanh(mean)
            return action, torch.zeros_like(value), value

        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        raw_action = dist.rsample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        action = torch.tanh(raw_action)
        # Enforce tanh squashing correction on log-prob
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1)
        return action, log_prob, value

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.actor(obs)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)

        # Invert tanh squashing safely
        eps = 1e-6
        clipped_actions = torch.clamp(actions, -1.0 + eps, 1.0 - eps)
        raw_actions = 0.5 * (torch.log1p(clipped_actions) - torch.log1p(-clipped_actions))

        log_prob = dist.log_prob(raw_actions).sum(dim=-1)
        log_prob -= torch.log(1.0 - clipped_actions.pow(2) + 1e-6).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        return log_prob, entropy, value

    def to_weights_dict(self) -> dict[str, Any]:
        """Convert frozen actor weights to dict format."""
        w1 = self.actor[0].weight.detach().cpu().numpy()  # (8, 6)
        b1 = self.actor[0].bias.detach().cpu().numpy()  # (8,)
        w2 = self.actor[2].weight.detach().cpu().numpy()  # (4, 8)
        b2 = self.actor[2].bias.detach().cpu().numpy()  # (4,)

        return {
            "name": "b5_ppo_actor",
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "params_count": self.count_actor_params(),
            "W1": w1.tolist(),
            "b1": b1.tolist(),
            "W2": w2.tolist(),
            "b2": b2.tolist(),
        }

    def export_weights(self, path: Path | str) -> dict[str, Any]:
        """Export frozen actor weights to JSON format for C99 embedded port."""
        data = self.to_weights_dict()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return data

    @classmethod
    def from_weights_dict(cls, data: dict[str, Any]) -> ActorCritic:
        model = cls(obs_dim=data.get("obs_dim", 6), act_dim=data.get("act_dim", 4))
        w1 = torch.tensor(data["W1"], dtype=torch.float32)
        b1 = torch.tensor(data["b1"], dtype=torch.float32)
        w2 = torch.tensor(data["W2"], dtype=torch.float32)
        b2 = torch.tensor(data["b2"], dtype=torch.float32)
        with torch.no_grad():
            model.actor[0].weight.copy_(w1)
            model.actor[0].bias.copy_(b1)
            model.actor[2].weight.copy_(w2)
            model.actor[2].bias.copy_(b2)
        return model
