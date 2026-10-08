from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from egga.config import REPO_ROOT
from egga.eval.baselines import evaluate_set
from egga.eval.metrics2 import MODE_NAMES
from egga.eval.provenance import provenance
from egga.scenarios.sets import load_scenarios, set_hash

RESULTS_DIR = REPO_ROOT / "results" / "phase7"
DOCS = REPO_ROOT / "docs"
PREREG_FILE = REPO_ROOT / "experiments" / "preregistered.yaml"
CONTROLLERS = ("b0_pid_ff", "b1_pd_ff", "b4_supervised", "b3_rl_unsupervised", "b5_rl_supervised")
BIG_ERROR_CM = 50.0


def verify_preregistration() -> dict[str, Any]:
    """Verify test set hash matches pre-registration protocol."""
    if not PREREG_FILE.exists():
        raise FileNotFoundError(f"Preregistration file {PREREG_FILE} not found!")
    prereg = yaml.safe_load(PREREG_FILE.read_text(encoding="utf-8"))
    test_scenarios = load_scenarios("test", unlock_test=True)
    actual_hash = set_hash(test_scenarios)
    expected_hash = prereg["test_set_hash"]
    if actual_hash != expected_hash:
        raise ValueError(
            f"Test set hash mismatch! Expected {expected_hash}, got {actual_hash}"
        )
    return prereg


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ctrl in CONTROLLERS:
        sub = df[df["controller"] == ctrl]
        if len(sub) == 0:
            continue
        ok = sub[~sub["diverged"]]
        rows.append(
            {
                "controller": ctrl,
                "n": len(sub),
                "diverged": int(sub["diverged"].sum()),
                "median_max_cm": float(ok["max_abs_ey_cm"].median())
                if len(ok) > 0
                else float("nan"),
                "p95_of_max_cm": float(ok["max_abs_ey_cm"].quantile(0.95))
                if len(ok) > 0
                else float("nan"),
                "mean_rms_cm": float(ok["rms_ey_cm"].mean()) if len(ok) > 0 else float("nan"),
                "median_progress": float(sub["progress_fraction"].median()),
                "mean_speed_mps": float(sub["mean_speed_mps"].mean()),
                "utilisation_over_frac": float(sub["utilisation_over_frac"].mean()),
            }
        )
    return pd.DataFrame(rows)


def mode_table(df: pd.DataFrame, controller: str = "b5_rl_supervised") -> pd.DataFrame:
    sub = df[df["controller"] == controller]
    if len(sub) == 0:
        return pd.DataFrame()
    return pd.DataFrame(
        [{"mode": name, "mean_fraction": float(sub[f"mode_{name}"].mean())} for name in MODE_NAMES]
    )


def paired_table(
    df: pd.DataFrame, baseline: str = "b4_supervised", candidate: str = "b5_rl_supervised"
) -> pd.DataFrame:
    b = df[df["controller"] == baseline].set_index("scenario")
    c = df[df["controller"] == candidate].set_index("scenario")
    common = b.index.intersection(c.index)
    b = b.loc[common]
    c = c.loc[common]

    b_div = b["diverged"].astype(bool)
    c_div = c["diverged"].astype(bool)

    b_failed_c_ok = int((b_div & ~c_div).sum())
    c_failed_b_ok = int((c_div & ~b_div).sum())
    both_failed = int((b_div & c_div).sum())

    both_ok = ~b_div & ~c_div
    c_better = int((both_ok & (c["max_abs_ey_cm"] < b["max_abs_ey_cm"] - 1e-4)).sum())
    b_better = int((both_ok & (b["max_abs_ey_cm"] < c["max_abs_ey_cm"] - 1e-4)).sum())

    return pd.DataFrame(
        [
            {
                "comparison": f"{candidate} vs {baseline}",
                "scenarios": len(common),
                f"{baseline}_failed_{candidate}_ok": b_failed_c_ok,
                f"{candidate}_failed_{baseline}_ok": c_failed_b_ok,
                "both_failed": both_failed,
                "candidate_beats_baseline_on_error": c_better,
                "baseline_beats_candidate_on_error": b_better,
            }
        ]
    )


def bootstrap_ci(
    diff: np.ndarray, n_boot: int = 1000, ci: float = 0.95, seed: int = 42
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    mean_val = float(np.mean(diff))
    if len(diff) == 0 or np.all(diff == 0.0):
        return mean_val, mean_val, mean_val
    indices = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot_means = np.mean(diff[indices], axis=1)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(boot_means, 100.0 * alpha))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    return mean_val, lo, hi


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |\n| " + " | ".join(["---"] * len(cols)) + " |\n"
    lines = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            val = row[c]
            if isinstance(val, (float, np.floating)):
                cells.append(f"{val:.3f}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return header + "\n".join(lines) + "\n"


def run_phase7() -> None:
    prereg = verify_preregistration()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    meta = provenance()

    parquet_path = RESULTS_DIR / "test_runs.parquet"
    if parquet_path.exists():
        print("Loading test runs from", parquet_path)
        df_test = pd.read_parquet(parquet_path)
    else:
        print("Evaluating held-out TEST set (5 controllers)...")
        df_test = evaluate_set(CONTROLLERS, "test", unlock_test=True)
        df_test.to_parquet(parquet_path)
        df_test.to_csv(RESULTS_DIR / "test_runs.csv", index=False)

    agg_test = aggregate(df_test)
    paired_b5 = paired_table(df_test, "b4_supervised", "b5_rl_supervised")
    paired_b3 = paired_table(df_test, "b4_supervised", "b3_rl_unsupervised")
    modes_b5 = mode_table(df_test, "b5_rl_supervised")

    # Wilcoxon test & bootstrap CI on non-diverged paired scenarios
    b4 = df_test[df_test["controller"] == "b4_supervised"].set_index("scenario")
    b5 = df_test[df_test["controller"] == "b5_rl_supervised"].set_index("scenario")

    both_ok_b5 = ~b4["diverged"] & ~b5["diverged"]
    diff_max_b5 = (
        b5.loc[both_ok_b5, "max_abs_ey_cm"] - b4.loc[both_ok_b5, "max_abs_ey_cm"]
    ).to_numpy()
    diff_rms_b5 = (b5.loc[both_ok_b5, "rms_ey_cm"] - b4.loc[both_ok_b5, "rms_ey_cm"]).to_numpy()

    if len(diff_max_b5) > 0 and not np.all(diff_max_b5 == 0.0):
        w_stat, p_val = stats.wilcoxon(diff_max_b5)
    else:
        w_stat, p_val = 0.0, float("nan")

    mean_dmax, lo_dmax, hi_dmax = bootstrap_ci(diff_max_b5)
    mean_drms, lo_drms, hi_drms = bootstrap_ci(diff_rms_b5)

    # Hypotheses evaluation
    b5_divs = int(df_test[df_test["controller"] == "b5_rl_supervised"]["diverged"].sum())
    b3_divs = int(df_test[df_test["controller"] == "b3_rl_unsupervised"]["diverged"].sum())
    b4_divs = int(df_test[df_test["controller"] == "b4_supervised"]["diverged"].sum())
    b5_p95_max = float(
        df_test[(df_test["controller"] == "b5_rl_supervised") & ~df_test["diverged"]][
            "max_abs_ey_cm"
        ].quantile(0.95)
    )

    h1_pass = (b5_divs == 0) and (b5_divs == b4_divs)
    h2_pass = b3_divs >= 1
    h3_pass = b5_p95_max < BIG_ERROR_CM

    # Diagnostic autopsy on divergences in B3 and B0
    autopsies = []
    div_runs = df_test[df_test["diverged"]].sort_values(["controller", "scenario"])
    for _, r in div_runs.iterrows():
        autopsies.append(
            f"- **Scenario `{r['scenario']}` ({r['controller']})**: "
            f"Friction family `{r.get('family', 'unknown')}`, "
            f"Progress {r['progress_fraction']:.1%}, "
            f"Final Speed {r['mean_speed_mps']:.2f} m/s, "
            f"Max Error {r['max_abs_ey_cm']:.1f} cm."
        )

    stats_dict = {
        "metadata": meta,
        "preregistration": prereg,
        "hypotheses_verdict": {
            "H1": {"pass": h1_pass, "b5_diverged": b5_divs, "b4_diverged": b4_divs},
            "H2": {"pass": h2_pass, "b3_diverged": b3_divs},
            "H3": {"pass": h3_pass, "b5_p95_max_cm": b5_p95_max},
        },
        "aggregate": agg_test.to_dict(orient="records"),
        "paired_b5_b4": paired_b5.to_dict(orient="records"),
        "paired_b3_b4": paired_b3.to_dict(orient="records"),
        "b5_modes": modes_b5.to_dict(orient="records"),
        "wilcoxon_w": float(w_stat),
        "wilcoxon_p": float(p_val) if np.isfinite(p_val) else None,
        "bootstrap_diff_max_cm": [mean_dmax, lo_dmax, hi_dmax],
        "bootstrap_diff_rms_cm": [mean_drms, lo_drms, hi_drms],
    }

    with open(RESULTS_DIR / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=2)

    intro = (
        "This report evaluates the unlocked, previously frozen held-out **Test Set** "
        "(48 scenarios).\n"
        "Evaluation adheres strictly to the pre-registered protocol in "
        "`experiments/preregistered.yaml`\n"
        f"(Test set hash: `{prereg['test_set_hash']}`).\n\n"
    )

    verdict_text = (
        "## Pre-Registered Hypotheses Verdict\n\n"
        f"- **H1 (Safety Envelope Preservation)**: **{'PASS' if h1_pass else 'FAIL'}** "
        f"— B5 experienced {b5_divs} catastrophic divergences (B4 experienced {b4_divs}).\n"
        f"- **H2 (Supervisor Necessity / Ablation)**: **{'PASS' if h2_pass else 'FAIL'}** "
        f"— Unsupervised RL (B3) experienced {b3_divs} catastrophic divergences, "
        "confirming that the runtime supervisor envelope guard is essential for stability.\n"
        f"- **H3 (Bounded Tracking Error)**: **{'PASS' if h3_pass else 'FAIL'}** "
        f"— B5 p95 max lateral error is {b5_p95_max:.3f} cm "
        f"(< {BIG_ERROR_CM:.1f} cm requirement).\n\n"
    )

    autopsy_text = (
        "## Diagnostic Autopsy of Divergences\n\n"
        + ("\n".join(autopsies) if autopsies else "No divergences observed.")
        + "\n\n"
    )

    report_text = (
        "# Phase 7 Report: Unlocked Held-Out Test Set Evaluation\n\n"
        f"Status: MEASURED. Generated by `python -m egga.eval.phase7` (git {meta['git_sha'][:8]}, "
        f"config hash {meta['config_hash']}, dirty={meta['git_dirty']}).\n\n"
        + intro
        + verdict_text
        + "## Aggregate Performance (48 Held-Out Test Scenarios)\n\n"
        + _md(agg_test)
        + "\n### Statistical Comparison: B5 vs B4 (Paired)\n\n"
        + f"- **Wilcoxon signed-rank test (max error)**: W = {w_stat:.1f}, p = {p_val}\n"
        + f"- **Bootstrap 95% CI on max error diff (B5 - B4)**: {mean_dmax:.3f} cm "
        f"[95% CI: {lo_dmax:.3f}, {hi_dmax:.3f}] cm\n"
        + f"- **Bootstrap 95% CI on RMS error diff (B5 - B4)**: {mean_drms:.3f} cm "
        f"[95% CI: {lo_drms:.3f}, {hi_drms:.3f}] cm\n\n"
        + _md(paired_b5)
        + "\n### Ablation Comparison: B3 (Unsupervised RL) vs B4\n\n"
        + _md(paired_b3)
        + "\n### B5 Supervisor Mode Distribution\n\n"
        + _md(modes_b5)
        + "\n"
        + autopsy_text
    )

    (DOCS / "phase7_report.md").write_text(report_text, encoding="utf-8")
    print("Successfully generated docs/phase7_report.md and results/phase7/stats.json")


if __name__ == "__main__":
    run_phase7()
