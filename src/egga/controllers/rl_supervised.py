from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from egga.controllers.base import Command, Observation, clip
from egga.controllers.classical import PIDFeedforwardController
from egga.controllers.derivative import FilteredDerivative
from egga.rl.env import DELTA_K_BOUNDS
from egga.rl.model import ActorCritic
from egga.supervisor.core import step as supervisor_step
from egga.supervisor.envelope import Envelope
from egga.supervisor.types import Config as SupConfig
from egga.supervisor.types import EstStatus, Inputs
from egga.supervisor.types import State as SupState

_STATUS = {
    "ok": int(EstStatus.OK),
    "low_excitation": int(EstStatus.LOW_EXCITATION),
    "stale": int(EstStatus.STALE),
    "invalid": int(EstStatus.INVALID),
}


class RLSupervisedController:
    """B5: RL-scheduled gains governed by the verified runtime supervisor and envelope guard."""

    name = "b5_rl_supervised"
    governs_speed = True

    def __init__(
        self,
        pid_cfg: dict[str, Any],
        wheelbase: float,
        vx: float,
        dt: float,
        cutoff_hz: float | None,
        sup_cfg: SupConfig,
        envelope: Envelope,
        model: ActorCritic | None = None,
        weights_path: Path | str | None = None,
    ) -> None:
        self._dt = dt
        self._cfg = sup_cfg
        self._env = envelope
        self._pid = PIDFeedforwardController(pid_cfg, dt, wheelbase, vx, cutoff_hz)
        self._rate = FilteredDerivative(cutoff_hz or 5.0, dt)
        ref = envelope.reference_gain
        self._reference = (float(ref[0]), float(ref[1]), float(ref[2]), float(ref[3]))

        if model is not None:
            self.model = model
        elif weights_path is not None:
            import json

            with open(weights_path, encoding="utf-8") as f:
                data = json.load(f)
            self.model = ActorCritic.from_weights_dict(data)
        else:
            self.model = ActorCritic()

        self._w1 = self.model.actor[0].weight.detach().cpu().numpy()
        self._b1 = self.model.actor[0].bias.detach().cpu().numpy()
        self._w2 = self.model.actor[2].weight.detach().cpu().numpy()
        self._b2 = self.model.actor[2].bias.detach().cpu().numpy()

        self.reset()

    def reset(self) -> None:
        self._pid.reset()
        self._rate.reset()
        self.state = SupState()
        self._prev_cmd = 0.0
        self._t = 0.0
        self._e_prev = 0.0
        self._k = 0

    def command(self, obs: Observation) -> Command:
        est = obs.estimate
        e_rate_meas = self._rate.update(obs.e_y)
        e_rate_abs = abs(e_rate_meas)

        if est is None:
            mu_lo, mu_hi, tau, mass_hi, quality, status = 0.1, 1.0, 0.15, 1.9, 0.0, 3
            mu_est = 0.55
            tau_est = 0.08
        else:
            mu_lo, mu_hi, tau = est.mu_lo, est.mu_hi, est.tau_bar
            mass_hi, quality, status = est.mass_hi, est.quality, _STATUS[est.status]
            mu_est = (mu_lo + mu_hi) * 0.5
            tau_est = tau

        de_fd = (obs.e_y - self._e_prev) / self._dt if self._k > 0 else 0.0
        self._e_prev = obs.e_y
        self._k += 1

        # 6-dim normalized observation vector strictly using estimates and measurements
        f1 = float(np.clip(obs.e_y / 0.5, -5.0, 5.0))
        f2 = float(np.clip(de_fd / 1.5, -5.0, 5.0))
        f3 = float(np.clip(obs.heading_error / 0.15, -5.0, 5.0))
        f4 = float(np.clip((obs.yaw_rate - obs.yaw_rate_ref) / 0.2, -5.0, 5.0))
        f5 = float(np.clip((mu_est - 0.55) / 0.35, -5.0, 5.0))
        f6 = float(np.clip((tau_est - 0.08) / 0.06, -5.0, 5.0))

        x = np.array([f1, f2, f3, f4, f5, f6], dtype=np.float32)
        h = np.tanh(self._w1 @ x + self._b1)
        act_np = np.tanh(self._w2 @ h + self._b2)

        delta_k = np.clip(act_np, -1.0, 1.0) * DELTA_K_BOUNDS
        rl_proposal = (
            float(self._reference[0] + delta_k[0]),
            float(self._reference[1] + delta_k[1]),
            float(self._reference[2] + delta_k[2]),
            float(self._reference[3] + delta_k[3]),
        )

        speed = obs.vx if obs.vx > 0.0 else 1.0
        inputs = Inputs(
            t=self._t,
            speed=speed,
            speed_request=obs.speed_request if obs.speed_request > 0.0 else speed,
            curvature_ahead=obs.curvature_ahead,
            mu_lo=mu_lo,
            mu_hi=mu_hi,
            tau_bar=tau,
            mass_hi=mass_hi,
            quality=quality,
            est_status=status,
            yaw_rate=obs.yaw_rate,
            steer_meas=obs.steer_meas,
            steer_cmd=self._prev_cmd,
            e_abs=abs(obs.e_y),
            e_rate_abs=e_rate_abs,
            rl_valid=True,
            rl_gain=rl_proposal,
            reference_gain=self._reference,
        )
        self._t += self._dt
        out = supervisor_step(self.state, self._cfg, self._env, inputs)

        self._pid.set_gains(*out.gains)
        raw = self._pid.command(obs).steer
        cmd = clip(raw, -out.steer_limit, out.steer_limit)
        self._prev_cmd = cmd
        return Command(steer=cmd, gains=out.gains, speed_cmd=out.speed_cmd, mode=out.mode)


class RLUnsupervisedController:
    """B3: RL-scheduled gains without supervisor.

    Ablation demonstrating failure under delay and noise without envelope guard.
    """

    name = "b3_rl_unsupervised"
    governs_speed = False

    def __init__(
        self,
        pid_cfg: dict[str, Any],
        wheelbase: float,
        vx: float,
        dt: float,
        cutoff_hz: float | None,
        reference_gain: tuple[float, float, float, float],
        model: ActorCritic | None = None,
        weights_path: Path | str | None = None,
    ) -> None:
        self._dt = dt
        self._pid = PIDFeedforwardController(pid_cfg, dt, wheelbase, vx, cutoff_hz)
        self._reference = reference_gain

        if model is not None:
            self.model = model
        elif weights_path is not None:
            import json

            with open(weights_path, encoding="utf-8") as f:
                data = json.load(f)
            self.model = ActorCritic.from_weights_dict(data)
        else:
            self.model = ActorCritic()

        self._w1 = self.model.actor[0].weight.detach().cpu().numpy()
        self._b1 = self.model.actor[0].bias.detach().cpu().numpy()
        self._w2 = self.model.actor[2].weight.detach().cpu().numpy()
        self._b2 = self.model.actor[2].bias.detach().cpu().numpy()

        self.reset()

    def reset(self) -> None:
        self._pid.reset()
        self._e_prev = 0.0
        self._k = 0

    def command(self, obs: Observation) -> Command:
        est = obs.estimate
        if est is None:
            mu_est, tau_est = 0.55, 0.08
        else:
            mu_est = (est.mu_lo + est.mu_hi) * 0.5
            tau_est = est.tau_bar

        de_fd = (obs.e_y - self._e_prev) / self._dt if self._k > 0 else 0.0
        self._e_prev = obs.e_y
        self._k += 1

        f1 = float(np.clip(obs.e_y / 0.5, -5.0, 5.0))
        f2 = float(np.clip(de_fd / 1.5, -5.0, 5.0))
        f3 = float(np.clip(obs.heading_error / 0.15, -5.0, 5.0))
        f4 = float(np.clip((obs.yaw_rate - obs.yaw_rate_ref) / 0.2, -5.0, 5.0))
        f5 = float(np.clip((mu_est - 0.55) / 0.35, -5.0, 5.0))
        f6 = float(np.clip((tau_est - 0.08) / 0.06, -5.0, 5.0))

        x = np.array([f1, f2, f3, f4, f5, f6], dtype=np.float32)
        h = np.tanh(self._w1 @ x + self._b1)
        act_np = np.tanh(self._w2 @ h + self._b2)

        delta_k = np.clip(act_np, -1.0, 1.0) * DELTA_K_BOUNDS
        applied_gains = (
            float(self._reference[0] + delta_k[0]),
            float(self._reference[1] + delta_k[1]),
            float(self._reference[2] + delta_k[2]),
            float(self._reference[3] + delta_k[3]),
        )

        self._pid.set_gains(*applied_gains)
        raw = self._pid.command(obs).steer
        # Full hardware limit, no supervisor guard
        cmd = clip(raw, -0.5, 0.5)
        return Command(steer=cmd, gains=applied_gains, speed_cmd=None, mode=0)
