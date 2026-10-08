from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from egga.config import REPO_ROOT, load_config, load_estimators
from egga.controllers.rl_supervised import RLSupervisedController, RLUnsupervisedController
from egga.estimation.suite import EstimatorSuite
from egga.eval.closed_loop import load_plant_config, run_closed_loop
from egga.rl.env import EGGAEnv
from egga.rl.model import ActorCritic
from egga.rl.ppo import PPOConfig, PPOTrainer
from egga.scenarios.sets import load_scenarios, run_arguments
from egga.supervisor.envelope import Envelope

logger = logging.getLogger(__name__)


def train_single_seed(
    seed: int,
    supervisor_enabled: bool = True,
    total_timesteps: int = 3000,
    n_steps: int = 500,
    batch_size: int = 64,
    n_epochs: int = 4,
    lr: float = 3e-4,
) -> tuple[ActorCritic, list[dict[str, float]]]:
    """Train a single RL policy seed."""
    torch.manual_seed(seed)

    env = EGGAEnv(scenario_set="train", supervisor_enabled=supervisor_enabled)
    model = ActorCritic()
    cfg = PPOConfig(
        total_timesteps=total_timesteps,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=lr,
    )
    trainer = PPOTrainer(env, model, cfg)
    metrics = trainer.train()
    return model, metrics


def evaluate_policy_on_scenarios(
    model: ActorCritic,
    scenario_set: str = "val",
    supervisor_enabled: bool = True,
    scenario_indices: list[int] | None = None,
) -> dict[str, float]:
    """Evaluate a trained policy across scenarios in scenario_set."""
    scenarios = load_scenarios(scenario_set)
    if scenario_indices is not None:
        scenarios = [scenarios[i] for i in scenario_indices if i < len(scenarios)]
    vehicle_cfg = load_config("vehicle.yaml")
    sup_cfg = None
    envelope = None
    from egga.supervisor.types import Config as SupConfig

    if supervisor_enabled:
        sup_cfg = SupConfig.from_dict(load_config("supervisor.yaml"), vehicle_cfg)
        envelope = Envelope.load(
            REPO_ROOT / "experiments" / "envelope" / "envelope_v1.npz",
            REPO_ROOT / "experiments" / "envelope" / "envelope_v1.json",
        )
    else:
        envelope = Envelope.load(
            REPO_ROOT / "experiments" / "envelope" / "envelope_v1.npz",
            REPO_ROOT / "experiments" / "envelope" / "envelope_v1.json",
        )

    ref = (
        float(envelope.reference_gain[0]),
        float(envelope.reference_gain[1]),
        float(envelope.reference_gain[2]),
        float(envelope.reference_gain[3]),
    )
    from egga.config import load_baselines

    pid_cfg = load_baselines()["pid_ff"]

    max_errors_cm: list[float] = []
    rms_errors_cm: list[float] = []
    diverged_count = 0

    for spec in scenarios:
        overrides, mission_cfg, belief_error, seed = run_arguments(spec)
        cfg = load_plant_config(overrides)
        dt = float(mission_cfg["dt_s"])
        wheelbase = float(vehicle_cfg["lf_m"]) + float(vehicle_cfg["lr_m"])
        speed = float(mission_cfg["vx_mps"]) if "vx_mps" in mission_cfg else 15.0

        estimator = EstimatorSuite(vehicle_cfg, load_estimators(), dt)

        if supervisor_enabled and sup_cfg is not None:
            ctrl = RLSupervisedController(
                pid_cfg=pid_cfg,
                wheelbase=wheelbase,
                vx=speed,
                dt=dt,
                cutoff_hz=5.0,
                sup_cfg=sup_cfg,
                envelope=envelope,
                model=model,
            )
        else:
            ctrl = RLUnsupervisedController(
                pid_cfg=pid_cfg,
                wheelbase=wheelbase,
                vx=speed,
                dt=dt,
                cutoff_hz=5.0,
                reference_gain=ref,
                model=model,
            )

        run = run_closed_loop(
            controller_name="b5_rl_supervised" if supervisor_enabled else "b3_rl_unsupervised",
            plant_cfg=cfg,
            seed=seed,
            mu_belief_error=belief_error,
            mission_cfg=mission_cfg,
            estimator=estimator,
            controller_instance=ctrl,
        )

        if run.diverged_at_s is not None:
            diverged_count += 1
        else:
            valid_ey = run.ey[np.isfinite(run.ey)]
            if len(valid_ey) > 0:
                max_errors_cm.append(float(np.max(np.abs(valid_ey)) * 100.0))
                rms_errors_cm.append(float(np.sqrt(np.mean(valid_ey**2)) * 100.0))

    med_max = float(np.median(max_errors_cm)) if max_errors_cm else float("nan")
    p95_max = float(np.percentile(max_errors_cm, 95)) if max_errors_cm else float("nan")
    mean_rms = float(np.mean(rms_errors_cm)) if rms_errors_cm else float("nan")

    return {
        "n_scenarios": float(len(scenarios)),
        "diverged": float(diverged_count),
        "median_max_cm": med_max,
        "p95_max_cm": p95_max,
        "mean_rms_cm": mean_rms,
    }


def run_training_campaign(
    n_seeds: int = 20,
    total_timesteps: int = 3000,
    output_dir: Path | None = None,
) -> None:
    """Run full 20-seed training for B5 (supervised) and B3 (unsupervised)."""
    out = output_dir or (REPO_ROOT / "results" / "phase6")
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []

    best_b5_model: ActorCritic | None = None
    best_b5_score = float("inf")
    best_b5_seed = -1

    best_b3_model: ActorCritic | None = None
    best_b3_score = float("inf")
    best_b3_seed = -1

    print(f"=== Starting Training Campaign: {n_seeds} seeds, {total_timesteps} steps/seed ===")

    probe_indices = [0, 4, 8, 12, 16]

    # 1. Train B5 (Supervised RL)
    print("\n--- Training B5 (Envelope-Guarded RL) ---", flush=True)
    for seed in range(n_seeds):
        model, metrics = train_single_seed(
            seed=seed,
            supervisor_enabled=True,
            total_timesteps=total_timesteps,
        )
        final_loss = metrics[-1]["policy_loss"] if metrics else 0.0
        final_rew = metrics[-1]["mean_reward"] if metrics else 0.0

        # Save checkpoint
        ckpt_path = ckpt_dir / f"b5_seed_{seed}.json"
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump(model.to_weights_dict(), f, indent=2)

        # Quick probe evaluation on validation set
        eval_res = evaluate_policy_on_scenarios(
            model, scenario_set="val", supervisor_enabled=True, scenario_indices=probe_indices
        )
        print(
            f"B5 Seed {seed:02d} | Train Rew: {final_rew:6.1f} | "
            f"Probe Div: {int(eval_res['diverged'])} | "
            f"Probe Med: {eval_res['median_max_cm']:.2f} cm | "
            f"Probe p95: {eval_res['p95_max_cm']:.2f} cm",
            flush=True,
        )

        score = eval_res["p95_max_cm"] + 1000.0 * eval_res["diverged"]
        if score < best_b5_score:
            best_b5_score = score
            best_b5_model = model
            best_b5_seed = seed

        summary_rows.append(
            {
                "controller": "b5_rl_supervised",
                "seed": seed,
                "train_reward": final_rew,
                "train_policy_loss": final_loss,
                "probe_diverged": int(eval_res["diverged"]),
                "probe_median_max_cm": eval_res["median_max_cm"],
                "probe_p95_max_cm": eval_res["p95_max_cm"],
                "probe_mean_rms_cm": eval_res["mean_rms_cm"],
            }
        )

    # 2. Train B3 (Unsupervised RL - Ablation)
    print("\n--- Training B3 (Unsupervised RL - Ablation) ---", flush=True)
    for seed in range(n_seeds):
        model, metrics = train_single_seed(
            seed=seed,
            supervisor_enabled=False,
            total_timesteps=total_timesteps,
        )
        final_loss = metrics[-1]["policy_loss"] if metrics else 0.0
        final_rew = metrics[-1]["mean_reward"] if metrics else 0.0

        ckpt_path = ckpt_dir / f"b3_seed_{seed}.json"
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump(model.to_weights_dict(), f, indent=2)

        eval_res = evaluate_policy_on_scenarios(
            model, scenario_set="val", supervisor_enabled=False, scenario_indices=probe_indices
        )
        print(
            f"B3 Seed {seed:02d} | Train Rew: {final_rew:6.1f} | "
            f"Probe Div: {int(eval_res['diverged'])} | "
            f"Probe Med: {eval_res['median_max_cm']:.2f} cm | "
            f"Probe p95: {eval_res['p95_max_cm']:.2f} cm",
            flush=True,
        )

        score = eval_res["p95_max_cm"] + 1000.0 * eval_res["diverged"]
        if score < best_b3_score:
            best_b3_score = score
            best_b3_model = model
            best_b3_seed = seed

        summary_rows.append(
            {
                "controller": "b3_rl_unsupervised",
                "seed": seed,
                "train_reward": final_rew,
                "train_policy_loss": final_loss,
                "probe_diverged": int(eval_res["diverged"]),
                "probe_median_max_cm": eval_res["median_max_cm"],
                "probe_p95_max_cm": eval_res["p95_max_cm"],
                "probe_mean_rms_cm": eval_res["mean_rms_cm"],
            }
        )

    # Save best models
    assert best_b5_model is not None
    b5_best_path = out / "b5_best.json"
    with open(b5_best_path, "w", encoding="utf-8") as f:
        json.dump(best_b5_model.to_weights_dict(), f, indent=2)
    print(f"\nBest B5 Model: Seed {best_b5_seed} saved to {b5_best_path}", flush=True)

    assert best_b3_model is not None
    b3_best_path = out / "b3_best.json"
    with open(b3_best_path, "w", encoding="utf-8") as f:
        json.dump(best_b3_model.to_weights_dict(), f, indent=2)
    print(f"Best B3 Model: Seed {best_b3_seed} saved to {b3_best_path}", flush=True)

    # Full validation of best models
    print("\n--- Full Validation Evaluation of Best Policies (24 scenarios) ---", flush=True)
    val_b5 = evaluate_policy_on_scenarios(
        best_b5_model, scenario_set="val", supervisor_enabled=True
    )
    b5_div = int(val_b5["diverged"])
    b5_med = val_b5["median_max_cm"]
    b5_p95 = val_b5["p95_max_cm"]
    print(
        f"Best B5 Full Val: Div {b5_div}/24 | Med {b5_med:.2f} cm | p95 {b5_p95:.2f} cm",
        flush=True,
    )

    val_b3 = evaluate_policy_on_scenarios(
        best_b3_model, scenario_set="val", supervisor_enabled=False
    )
    b3_div = int(val_b3["diverged"])
    b3_med = val_b3["median_max_cm"]
    b3_p95 = val_b3["p95_max_cm"]
    print(
        f"Best B3 Full Val: Div {b3_div}/24 | Med {b3_med:.2f} cm | p95 {b3_p95:.2f} cm",
        flush=True,
    )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out / "training_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Training summary saved to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 20 seeds of B5 and B3 RL policies.")
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds to train")
    parser.add_argument("--timesteps", type=int, default=3000, help="Timesteps per seed")
    parser.add_argument("--out", type=str, default=str(REPO_ROOT / "results" / "phase6"))
    args = parser.parse_args()

    run_training_campaign(
        n_seeds=args.seeds, total_timesteps=args.timesteps, output_dir=Path(args.out)
    )


if __name__ == "__main__":
    main()
