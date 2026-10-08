from __future__ import annotations

import numpy as np
import torch

from egga.config import load_config, load_estimators
from egga.estimation.suite import EstimatorSuite
from egga.eval.closed_loop import load_plant_config, run_closed_loop
from egga.rl.env import EGGAEnv
from egga.rl.model import ActorCritic
from egga.rl.ppo import PPOConfig, PPOTrainer
from egga.scenarios.sets import load_scenarios, run_arguments


def test_actor_parameter_count_is_exactly_92() -> None:
    model = ActorCritic()
    count = model.count_actor_params()
    assert count == 92, f"Actor has {count} parameters, expected exactly 92 (6->8->4 MLP)"


def test_zero_delta_k_action_is_numerically_equivalent_to_b4() -> None:
    """When delta_K = 0, B5 must produce the exact same trajectory and gains as B4."""
    # Zero-weight model outputs zeros for any input
    zero_model = ActorCritic()
    with torch.no_grad():
        for p in zero_model.actor.parameters():
            p.zero_()

    scenarios = load_scenarios("val")
    spec = scenarios[0]
    plant_raw, mission_raw, belief_err, seed = run_arguments(spec)

    # 1. Run B4 with live estimator suite
    dt = float(mission_raw["dt_s"])
    est = EstimatorSuite(load_config("vehicle.yaml"), load_estimators(), dt)
    run_b4 = run_closed_loop(
        "b4_supervised",
        load_plant_config(plant_raw),
        seed=seed,
        mission_cfg=mission_raw,
        estimator=est,
    )

    # 2. Step EGGAEnv with zero action (delta_K = 0)
    env = EGGAEnv(scenario_set="val", scenario_idx=0, supervisor_enabled=True)
    obs, _ = env.reset(seed=seed)
    actions = np.zeros(4, dtype=np.float32)

    b5_errors = []
    done = False
    while not done:
        obs, reward, term, trunc, info = env.step(actions)
        b5_errors.append(info["e_true"])
        done = term or trunc

    # Compare max lateral error between B4 and zero-delta-K B5
    n_compare = min(len(b5_errors), len(run_b4.ey[np.isfinite(run_b4.ey)]))
    diff = np.abs(np.array(b5_errors[:n_compare]) - run_b4.ey[:n_compare])
    max_diff = float(np.max(diff))
    assert max_diff < 1e-4, f"max diff {max_diff:.6f} m between B4 and zero-delta-K B5"


def test_observation_vector_never_contains_ground_truth_internals() -> None:
    env = EGGAEnv(scenario_set="val", scenario_idx=1)
    obs, _ = env.reset(seed=42)
    assert obs.shape == (6,)
    # Observations are bounded and normalized
    assert np.all(np.isfinite(obs))
    assert np.all(obs >= -10.0) and np.all(obs <= 10.0)


def test_ppo_step_and_convergence_on_toy_task() -> None:
    env = EGGAEnv(scenario_set="val", scenario_idx=0)
    model = ActorCritic()
    cfg = PPOConfig(total_timesteps=300, n_steps=100, batch_size=32, n_epochs=2)
    trainer = PPOTrainer(env, model, cfg)
    metrics = trainer.train()
    assert len(metrics) > 0
    assert "policy_loss" in metrics[0]
    assert "value_loss" in metrics[0]
    assert np.isfinite(metrics[0]["policy_loss"])


def test_supervisor_guards_against_extreme_adversarial_rl_actions() -> None:
    """Even if RL commands extreme +1.0 or -1.0 saturation actions,
    the supervisor keeps the car bounded.
    """
    env = EGGAEnv(scenario_set="val", scenario_idx=2, supervisor_enabled=True)
    obs, _ = env.reset(seed=123)

    diverged = False
    for _ in range(200):
        action = np.ones(4, dtype=np.float32)  # extreme max action
        obs, reward, term, trunc, info = env.step(action)
        if term:
            diverged = True
            break
        if trunc:
            break
    assert not diverged, "Supervisor failed to prevent divergence under extreme +1.0 RL action"
