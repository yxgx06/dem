from __future__ import annotations

import copy
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from egga.config import load_config
from egga.controllers.base import clip
from egga.controllers.classical import PIDFeedforwardController
from egga.controllers.derivative import FilteredDerivative
from egga.estimation.suite import EstimatorSuite
from egga.estimation.types import Measurements
from egga.eval.baselines import load_estimators
from egga.eval.closed_loop import (
    MU_BELIEF_CEIL,
    MU_BELIEF_FLOOR,
    _wrap,
    load_plant_config,
    speed_profile,
)
from egga.eval.envelope_build import ENVELOPE_DIR
from egga.plant.actuator import SteeringActuator
from egga.plant.friction import FrictionProfile
from egga.plant.sensors import Sensor
from egga.plant.vehicle import PlantParams, Vehicle
from egga.scenarios.mission import build_mission
from egga.scenarios.sets import load_scenarios, run_arguments
from egga.supervisor.core import step as supervisor_step
from egga.supervisor.envelope import Envelope
from egga.supervisor.types import Config as SupConfig
from egga.supervisor.types import EstStatus, Inputs
from egga.supervisor.types import State as SupState

_STATUS_MAP = {
    "ok": int(EstStatus.OK),
    "low_excitation": int(EstStatus.LOW_EXCITATION),
    "stale": int(EstStatus.STALE),
    "invalid": int(EstStatus.INVALID),
}

# Delta-K bounds around reference gains: [delta_Kp, delta_Ki, delta_Kd, delta_Khead]
DELTA_K_BOUNDS = np.array([0.6, 0.04, 0.15, 0.6], dtype=np.float32)


class EGGAEnv(gym.Env):
    """Gymnasium environment for Envelope-Guarded Gain Adaptation (EGGA).

    Observation: 6-dimensional normalized vector:
      [e_y / 0.5, de_y / 1.5, heading_error / 0.15, yaw_error / 0.2,
       (mu_est - 0.5) / 0.35, (tau_bar - 0.08) / 0.06]
    Action: 4-dimensional continuous vector in [-1, 1]^4, scaled to delta-K around nominal.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario_set: str = "train",
        scenario_idx: int | None = None,
        supervisor_enabled: bool = True,
        oracle_estimates: bool = False,
    ) -> None:
        super().__init__()
        self.scenario_set = scenario_set
        self.fixed_scenario_idx = scenario_idx
        self.supervisor_enabled = supervisor_enabled
        self.oracle_estimates = oracle_estimates

        self.scenarios = load_scenarios(scenario_set) if scenario_set in ("train", "val") else []
        self.env_envelope = Envelope.load(
            ENVELOPE_DIR / "envelope_v1.npz", ENVELOPE_DIR / "envelope_v1.json"
        )
        self.vehicle_cfg = load_config("vehicle.yaml")
        self.sup_cfg = SupConfig.from_dict(load_config("supervisor.yaml"), self.vehicle_cfg)
        from egga.config import load_baselines

        self.pid_cfg = load_baselines()["pid_ff"]
        ref = self.env_envelope.reference_gain
        self.reference_gains = (float(ref[0]), float(ref[1]), float(ref[2]), float(ref[3]))

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(6,), dtype=np.float32)

        self._current_scenario_idx = 0
        self._step_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if options and "scenario_idx" in options:
            idx = int(options["scenario_idx"])
        elif self.fixed_scenario_idx is not None:
            idx = self.fixed_scenario_idx
        elif self.scenarios:
            idx = int(self.np_random.integers(0, len(self.scenarios)))
        else:
            idx = 0
        self._current_scenario_idx = idx

        spec = copy.deepcopy(self.scenarios[idx]) if self.scenarios else {}
        plant_raw, mission_cfg, belief_err, run_seed = run_arguments(spec)
        self.plant_cfg = load_plant_config(plant_raw)
        self.mu_belief_error = belief_err
        self.run_seed = run_seed if seed is None else seed

        self.dt = float(mission_cfg["dt_s"])
        n = int(round(float(mission_cfg["t_final_s"]) / self.dt)) + 1
        vx_t = speed_profile(self.plant_cfg, n, self.dt, mission_cfg)
        self.mission = build_mission(vx=vx_t, cfg=mission_cfg)
        self.n_steps = len(self.mission.t)
        self.k = 0
        self.path_s = 0.0
        self.ay_prev = 0.0
        self.e_prev = 0.0
        self.v_cur = float(self.mission.vx_t[0])

        params = PlantParams.from_configs(self.vehicle_cfg, self.plant_cfg)
        self.vehicle = Vehicle(
            params,
            x0=float(self.mission.x_ref[0]),
            y0=float(self.mission.y_ref[0]),
            psi0=float(self.mission.psi_ref[0]),
        )
        children = np.random.SeedSequence(self.run_seed).spawn(5)
        rngs = [np.random.default_rng(c) for c in children]
        self.sensors = {
            "lateral_error": Sensor(self.plant_cfg["sensors"]["lateral_error"], rngs[1]),
            "yaw_rate": Sensor(self.plant_cfg["sensors"]["yaw_rate"], rngs[2]),
            "steering_angle": Sensor(self.plant_cfg["sensors"]["steering_angle"], rngs[3]),
            "lateral_accel": Sensor(self.plant_cfg["sensors"]["lateral_accel"], rngs[4]),
        }
        self.actuator = SteeringActuator(
            self.plant_cfg["actuator"],
            self.dt,
            float(self.vehicle_cfg["steer_rate_max_rps"]),
            float(self.vehicle_cfg["delta_max_rad"]),
            rngs[0],
        )

        f_cfg = self.plant_cfg["friction"]
        if f_cfg["profile"] is None:
            self.friction = FrictionProfile.from_array(self.mission.mu, self.dt)
        else:
            self.friction = FrictionProfile.from_config(f_cfg["profile"], f_cfg["split_mu"])

        self.estimator = EstimatorSuite(self.vehicle_cfg, load_estimators(), self.dt)
        self.estimator.reset()
        self.last_estimate = None

        wheelbase = float(self.vehicle_cfg["lf_m"]) + float(self.vehicle_cfg["lr_m"])
        self.pid = PIDFeedforwardController(
            self.pid_cfg, self.dt, wheelbase, self.v_cur, derivative_cutoff_hz=5.0
        )
        self.pid.reset()
        self.rate_filter = FilteredDerivative(5.0, self.dt)
        self.rate_filter.reset()

        self.sup_state = SupState()
        self.prev_steer_cmd = 0.0
        self.t_sup = 0.0

        self.arclength = np.concatenate(
            [[0.0], np.cumsum(np.hypot(np.diff(self.mission.x_ref), np.diff(self.mission.y_ref)))]
        )
        self.curvature_ref = self.mission.r_ref / self.mission.vx_t

        obs = np.zeros(6, dtype=np.float32)
        info: dict[str, Any] = {"scenario_idx": idx, "seed": self.run_seed}
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        t_k = float(self.mission.t[self.k])
        s_v = self.path_s
        x_r = float(np.interp(s_v, self.arclength, self.mission.x_ref))
        y_r = float(np.interp(s_v, self.arclength, self.mission.y_ref))
        psi_r = float(np.interp(s_v, self.arclength, self.mission.psi_ref))
        vx_k = self.v_cur
        r_ref_k = float(np.interp(s_v, self.arclength, self.curvature_ref)) * vx_k
        slope_k = float(np.interp(s_v, self.arclength, self.mission.slope_deg))
        v_plan = float(np.interp(s_v, self.arclength, self.mission.vx_t))

        look = max(30.0, 3.0 * vx_k)
        j0 = int(np.searchsorted(self.arclength, s_v))
        j1 = int(np.searchsorted(self.arclength, s_v + look)) + 1
        curv_ahead = float(np.max(np.abs(self.curvature_ref[j0:j1]))) if j1 > j0 else 0.0

        ex_g = x_r - self.vehicle.x
        ey_g = y_r - self.vehicle.y
        e_true = float(ey_g * np.cos(self.vehicle.psi) - ex_g * np.sin(self.vehicle.psi))

        diverged = not np.isfinite(e_true) or abs(e_true) > 2.0
        if diverged:
            reward = -100.0
            return (
                np.zeros(6, dtype=np.float32),
                reward,
                True,
                False,
                {"diverged": True, "progress": s_v / self.arclength[-1]},
            )

        m_ey = self.sensors["lateral_error"].measure(e_true)
        m_r = self.sensors["yaw_rate"].measure(self.vehicle.r)
        m_st = self.sensors["steering_angle"].measure(self.actuator.angle)
        m_ay = self.sensors["lateral_accel"].measure(self.ay_prev)

        mu_l, mu_r = self.friction.mu_wheels(t_k, s_v)
        mu_true = self.friction.mu(t_k, s_v)
        belief = float(np.clip(mu_true + self.mu_belief_error, MU_BELIEF_FLOOR, MU_BELIEF_CEIL))

        de_fd = (m_ey.value - self.e_prev) / self.dt if self.k > 0 else 0.0
        self.e_prev = m_ey.value

        # Candidate gain from RL action: delta_K around reference
        delta_k = np.clip(action, -1.0, 1.0) * DELTA_K_BOUNDS
        rl_proposal = (
            float(self.reference_gains[0] + delta_k[0]),
            float(self.reference_gains[1] + delta_k[1]),
            float(self.reference_gains[2] + delta_k[2]),
            float(self.reference_gains[3] + delta_k[3]),
        )

        rl_rejected = False
        forced_jump = False
        if self.supervisor_enabled:
            est = self.last_estimate
            e_rate_abs = abs(self.rate_filter.update(m_ey.value))
            if est is None:
                mu_lo, mu_hi, tau, mass_hi, quality, status = 0.1, 1.0, 0.15, 1.9, 0.0, 3
            else:
                mu_lo, mu_hi, tau = est.mu_lo, est.mu_hi, est.tau_bar
                mass_hi, quality, status = est.mass_hi, est.quality, _STATUS_MAP[est.status]
            sup_inputs = Inputs(
                t=self.t_sup,
                speed=vx_k if vx_k > 0.0 else 1.0,
                speed_request=v_plan if v_plan > 0.0 else vx_k,
                curvature_ahead=curv_ahead,
                mu_lo=mu_lo,
                mu_hi=mu_hi,
                tau_bar=tau,
                mass_hi=mass_hi,
                quality=quality,
                est_status=status,
                yaw_rate=m_r.value,
                steer_meas=m_st.value,
                steer_cmd=self.prev_steer_cmd,
                e_abs=abs(m_ey.value),
                e_rate_abs=e_rate_abs,
                rl_valid=True,
                rl_gain=rl_proposal,
                reference_gain=self.reference_gains,
            )
            self.t_sup += self.dt
            sup_out = supervisor_step(self.sup_state, self.sup_cfg, self.env_envelope, sup_inputs)
            applied_gains = sup_out.gains
            speed_cmd = sup_out.speed_cmd
            steer_limit = sup_out.steer_limit
            rl_rejected = not sup_out.rl_applied
            forced_jump = sup_out.forced_gain_jump
            mode = sup_out.mode
        else:
            applied_gains = rl_proposal
            speed_cmd = v_plan
            steer_limit = float(self.vehicle_cfg["delta_max_rad"])
            mode = 0

        # Classical steering calculation with applied gains
        self.pid.set_gains(*applied_gains)
        obs_classical = type(
            "ClassicalObs",
            (),
            {
                "e_y": m_ey.value,
                "de_y": de_fd,
                "heading_error": _wrap(psi_r - self.vehicle.psi),
                "yaw_rate": m_r.value,
                "yaw_rate_ref": r_ref_k,
                "slope_deg": slope_k,
                "mu_belief": belief,
                "vx": vx_k,
                "r_ref_preview": (),
                "estimate": self.last_estimate,
                "steer_meas": m_st.value,
                "speed_request": v_plan,
                "curvature_ahead": curv_ahead,
            },
        )()
        raw_cmd = self.pid.command(obs_classical).steer
        steer_cmd = clip(raw_cmd, -steer_limit, steer_limit)
        self.prev_steer_cmd = steer_cmd

        # Estimator update
        self.last_estimate = self.estimator.update(
            Measurements(
                t=t_k,
                vx=vx_k,
                yaw_rate=m_r.value,
                yaw_rate_valid=m_r.valid,
                lateral_accel=m_ay.value,
                lateral_accel_valid=m_ay.valid,
                steer_meas=m_st.value,
                steer_meas_valid=m_st.valid,
                steer_cmd=steer_cmd,
            )
        )

        # Plant stepping
        delta = self.actuator.step(steer_cmd)
        v_new = max(float(speed_cmd), 1.0)
        ax_k = (v_new - vx_k) / self.dt
        self.v_cur = v_new
        veh_out = self.vehicle.step(delta, mu_l, mu_r, slope_k, self.dt, v_new, ax_k, t_k)
        self.path_s += v_new * self.dt
        self.ay_prev = veh_out.ay

        # Reward formulation with multi-component logging
        heading_err = _wrap(psi_r - self.vehicle.psi)
        r_track = -15.0 * (e_true**2) - 3.0 * (heading_err**2)
        r_effort = -0.5 * (delta**2)
        ay_max = float(self.sup_cfg.gravity * mu_true * 0.8)
        r_margin = -5.0 * max(0.0, abs(veh_out.ay) / max(ay_max, 1.0) - 0.8) ** 2
        r_robust = -2.0 * ((self.vehicle.r - r_ref_k) ** 2)
        p_override = (-1.0 if rl_rejected else 0.0) + (-2.0 if forced_jump else 0.0)
        p_envelope = -20.0 if abs(e_true) > 0.3 else 0.0
        total_reward = float(r_track + r_effort + r_margin + r_robust + p_override + p_envelope)

        self.k += 1
        terminated = False
        truncated = self.k >= self.n_steps - 1

        if self.last_estimate is not None:
            mu_est = (self.last_estimate.mu_lo + self.last_estimate.mu_hi) * 0.5
            tau_est = self.last_estimate.tau_bar
        else:
            mu_est = 0.55
            tau_est = 0.08

        f1 = float(np.clip(m_ey.value / 0.5, -5.0, 5.0))
        f2 = float(np.clip(de_fd / 1.5, -5.0, 5.0))
        f3 = float(np.clip(heading_err / 0.15, -5.0, 5.0))
        f4 = float(np.clip((m_r.value - r_ref_k) / 0.2, -5.0, 5.0))
        f5 = float(np.clip((mu_est - 0.55) / 0.35, -5.0, 5.0))
        f6 = float(np.clip((tau_est - 0.08) / 0.06, -5.0, 5.0))
        next_obs = np.array([f1, f2, f3, f4, f5, f6], dtype=np.float32)

        info = {
            "e_true": e_true,
            "progress": self.path_s / self.arclength[-1],
            "mode": mode,
            "reward_track": r_track,
            "reward_effort": r_effort,
            "reward_margin": r_margin,
            "reward_robust": r_robust,
            "penalty_override": p_override,
            "penalty_envelope": p_envelope,
            "rl_rejected": rl_rejected,
            "applied_gains": applied_gains,
        }
        return next_obs, total_reward, terminated, truncated, info
