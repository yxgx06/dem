from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from egga.config import REPO_ROOT
from egga.eval.baselines import evaluate_set
from egga.eval.metrics2 import MODE_NAMES
from egga.eval.provenance import provenance

RESULTS_DIR = REPO_ROOT / "results" / "phase6"
DOCS = REPO_ROOT / "docs"
CONTROLLERS = ("b0_pid_ff", "b1_pd_ff", "b4_supervised", "b3_rl_unsupervised", "b5_rl_supervised")
EVAL_SETS = ("train", "val")
BIG_ERROR_CM = 50.0


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
    """Pairwise comparison scenario by scenario."""
    b_df = df[df["controller"] == baseline].set_index("scenario")
    c_df = df[df["controller"] == candidate].set_index("scenario").loc[b_df.index]

    failed_b = b_df["diverged"] | (b_df["max_abs_ey_cm"] > BIG_ERROR_CM)
    failed_c = c_df["diverged"] | (c_df["max_abs_ey_cm"] > BIG_ERROR_CM)

    return pd.DataFrame(
        [
            {
                "comparison": f"{candidate} vs {baseline}",
                "scenarios": len(b_df),
                f"{baseline}_failed_{candidate}_ok": int((failed_b & ~failed_c).sum()),
                f"{candidate}_failed_{baseline}_ok": int((failed_c & ~failed_b).sum()),
                "both_failed": int((failed_b & failed_c).sum()),
                "candidate_beats_baseline_on_error": int(
                    (c_df["max_abs_ey_cm"] < b_df["max_abs_ey_cm"]).sum()
                ),
                "baseline_beats_candidate_on_error": int(
                    (b_df["max_abs_ey_cm"] < c_df["max_abs_ey_cm"]).sum()
                ),
            }
        ]
    )


def bootstrap_ci(
    diffs: np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap mean difference and two-sided confidence interval."""
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    n = len(diffs)
    for i in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    lo = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return float(np.mean(diffs)), lo, hi


def compute_statistical_tests(df: pd.DataFrame) -> dict[str, Any]:
    """Compute Wilcoxon signed-rank and bootstrap CIs between B5 and B4."""
    b4 = df[df["controller"] == "b4_supervised"].set_index("scenario")
    b5 = df[df["controller"] == "b5_rl_supervised"].set_index("scenario").loc[b4.index]

    # Max lateral error differences (B5 - B4 in cm)
    e_b4 = b4["max_abs_ey_cm"].to_numpy()
    e_b5 = b5["max_abs_ey_cm"].to_numpy()
    diff_max = e_b5 - e_b4

    # RMS error differences (B5 - B4 in cm)
    rms_b4 = b4["rms_ey_cm"].to_numpy()
    rms_b5 = b5["rms_ey_cm"].to_numpy()
    diff_rms = rms_b5 - rms_b4

    # Wilcoxon signed-rank test
    w_stat, w_p = stats.wilcoxon(diff_max, zero_method="wilcox", alternative="two-sided")
    w_rms_stat, w_rms_p = stats.wilcoxon(diff_rms, zero_method="wilcox", alternative="two-sided")

    # Bootstrap 95% CIs
    mean_diff_max, ci_max_lo, ci_max_hi = bootstrap_ci(diff_max)
    mean_diff_rms, ci_rms_lo, ci_rms_hi = bootstrap_ci(diff_rms)

    return {
        "n_scenarios": len(b4),
        "wilcoxon_max_stat": float(w_stat),
        "wilcoxon_max_pvalue": float(w_p),
        "wilcoxon_rms_stat": float(w_rms_stat),
        "wilcoxon_rms_pvalue": float(w_rms_p),
        "mean_diff_max_cm": mean_diff_max,
        "ci95_diff_max_cm": [ci_max_lo, ci_max_hi],
        "mean_diff_rms_cm": mean_diff_rms,
        "ci95_diff_rms_cm": [ci_rms_lo, ci_rms_hi],
        "median_b4_max_cm": float(np.median(e_b4)),
        "median_b5_max_cm": float(np.median(e_b5)),
        "p95_b4_max_cm": float(np.percentile(e_b4, 95)),
        "p95_b5_max_cm": float(np.percentile(e_b5, 95)),
    }


def _md(df: pd.DataFrame, floats: int = 3) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        cells = [f"{v:.{floats}f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run_phase6_evaluation() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    meta = provenance()

    sections: list[str] = []
    stats_dict: dict[str, Any] = {"provenance": meta}

    for set_name in EVAL_SETS:
        print(f"Evaluating {set_name} set ({len(CONTROLLERS)} controllers)...", flush=True)
        df = evaluate_set(CONTROLLERS, set_name)
        df.to_parquet(RESULTS_DIR / f"{set_name}_runs.parquet", index=False)
        df.to_csv(RESULTS_DIR / f"{set_name}_runs.csv", index=False)

        agg = aggregate(df)
        paired_b5_b4 = paired_table(df, "b4_supervised", "b5_rl_supervised")
        paired_b3_b4 = paired_table(df, "b4_supervised", "b3_rl_unsupervised")
        modes_b5 = mode_table(df, "b5_rl_supervised")
        st = compute_statistical_tests(df)
        stats_dict[set_name] = st

        w_stat = st["wilcoxon_max_stat"]
        w_p = st["wilcoxon_max_pvalue"]
        m_diff = st["mean_diff_max_cm"]
        ci_lo, ci_hi = st["ci95_diff_max_cm"]
        rms_diff = st["mean_diff_rms_cm"]
        rms_lo, rms_hi = st["ci95_diff_rms_cm"]

        stat_block = (
            f"\n\n### Statistical Comparison: B5 vs B4 (Paired)\n\n"
            f"- **Wilcoxon signed-rank test (max error)**: W = {w_stat:.1f}, p = {w_p:.4e}\n"
            f"- **Bootstrap 95% CI on max error diff (B5 - B4)**: {m_diff:.3f} cm "
            f"[95% CI: {ci_lo:.3f}, {ci_hi:.3f}] cm\n"
            f"- **Bootstrap 95% CI on RMS error diff (B5 - B4)**: {rms_diff:.3f} cm "
            f"[95% CI: {rms_lo:.3f}, {rms_hi:.3f}] cm\n\n"
        )

        sections.append(
            f"## {set_name.capitalize()} Set ({df['scenario'].nunique()} scenarios)\n\n"
            f"### Aggregate Performance\n\n"
            + _md(agg)
            + stat_block
            + _md(paired_b5_b4)
            + "\n\n### Ablation Comparison: B3 (Unsupervised RL) vs B4\n\n"
            + _md(paired_b3_b4)
            + "\n\n### B5 Supervisor Mode Distribution\n\n"
            + _md(modes_b5)
        )

    with open(RESULTS_DIR / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=2)

    intro = (
        "B5 uses a 92-parameter MLP policy proposing gain adjustments bounded around nominal "
        "gains,\n"
        "screened and envelope-guarded by the verified runtime supervisor. B3 is the ablation\n"
        "controller commanding gains directly without the supervisor envelope guard.\n\n"
    )

    report_text = (
        "# Phase 6 Report: B5 (Envelope-Guarded RL) vs B4 and Ablation B3\n\n"
        f"Status: MEASURED. Generated by `python -m egga.eval.phase6` (git {meta['git_sha'][:8]}, "
        f"config hash {meta['config_hash']}, dirty={meta['git_dirty']}).\n\n"
        + intro
        + "\n\n".join(sections)
        + "\n"
    )

    (DOCS / "phase6_report.md").write_text(report_text, encoding="utf-8")
    print("Successfully generated docs/phase6_report.md and results/phase6/stats.json")


def main() -> None:
    run_phase6_evaluation()


if __name__ == "__main__":
    main()
