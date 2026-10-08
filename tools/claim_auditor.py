from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

from egga.config import REPO_ROOT

DOCS_DIR = REPO_ROOT / "docs"
RESULTS_DIR = REPO_ROOT / "results"
SAFETY_DIR = DOCS_DIR / "safety"


class ClaimAuditor:
    def __init__(self) -> None:
        self.total_checks = 0
        self.passed_checks = 0
        self.failures: list[str] = []

    def check(self, condition: bool, description: str) -> None:
        self.total_checks += 1
        if condition:
            self.passed_checks += 1
            print(f"  [PASS] {description}")
        else:
            self.failures.append(description)
            print(f"  [FAIL] {description}")

    def audit_traceability(self) -> None:
        print("\n--- Auditing Safety Traceability Matrix (docs/safety/traceability.csv) ---")
        csv_path = SAFETY_DIR / "traceability.csv"
        self.check(csv_path.exists(), f"File exists: {csv_path.relative_to(REPO_ROOT)}")
        if not csv_path.exists():
            return

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.check(len(rows) >= 10, f"Found {len(rows)} traceability requirements (expected >= 10)")

        for r in rows:
            req_id = r.get("Requirement_ID", "")
            asil = r.get("ASIL", "")
            sg_id = r.get("Safety_Goal_ID", "")
            py_impl = r.get("Implementation_Python", "").split(":")[0]
            c_impl = r.get("Implementation_C99", "").split(":")[0]
            test_file = r.get("Test_File", "").split(":")[0]

            self.check(
                asil in ("ASIL A", "ASIL B", "ASIL C", "ASIL D"),
                f"{req_id}: Valid ASIL rating ({asil})",
            )
            self.check(
                bool(re.match(r"^SG-\d{2}$", sg_id)),
                f"{req_id}: Valid Safety Goal ({sg_id})",
            )

            py_path = REPO_ROOT / py_impl
            self.check(py_path.exists(), f"{req_id}: Python implementation exists ({py_impl})")

            c_path = REPO_ROOT / c_impl
            self.check(c_path.exists(), f"{req_id}: C99 implementation exists ({c_impl})")

            t_path = REPO_ROOT / test_file
            self.check(t_path.exists(), f"{req_id}: Verification test file exists ({test_file})")

    def audit_sotif_triggers(self) -> None:
        print("\n--- Auditing SOTIF Triggering Conditions (docs/safety/sotif_triggers.csv) ---")
        csv_path = SAFETY_DIR / "sotif_triggers.csv"
        self.check(csv_path.exists(), f"File exists: {csv_path.relative_to(REPO_ROOT)}")
        if not csv_path.exists():
            return

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.check(
            len(rows) >= 6, f"Found {len(rows)} SOTIF triggering conditions (expected >= 6)"
        )
        for r in rows:
            trig_id = r.get("Trigger_ID", "")
            self.check(
                bool(re.match(r"^SOTIF-TC-\d{2}$", trig_id)), f"Valid Trigger ID: {trig_id}"
            )
            self.check(
                len(r.get("Mitigation_Mechanism", "")) > 10, f"{trig_id}: Mitigation defined"
            )
            self.check(
                len(r.get("Supervisor_Action", "")) > 10, f"{trig_id}: Supervisor action defined"
            )

    def audit_phase8_claims(self) -> None:
        print("\n--- Auditing Phase 8 Claims (C99 Embedded Benchmark & Equivalence) ---")
        bench_path = RESULTS_DIR / "phase8" / "c99_benchmark.json"
        equiv_path = RESULTS_DIR / "phase8" / "c99_equivalence.json"
        report_path = DOCS_DIR / "phase8_report.md"

        self.check(bench_path.exists(), "results/phase8/c99_benchmark.json exists")
        self.check(equiv_path.exists(), "results/phase8/c99_equivalence.json exists")
        self.check(report_path.exists(), "docs/phase8_report.md exists")

        if not (bench_path.exists() and equiv_path.exists() and report_path.exists()):
            return

        with open(bench_path, encoding="utf-8") as f:
            bench = json.load(f)
        with open(equiv_path, encoding="utf-8") as f:
            equiv = json.load(f)
        report_text = report_path.read_text(encoding="utf-8")

        # Memory claims
        self.check(
            f"{bench['sizeof_state_bytes']:,} B" in report_text,
            f"State struct size grounded: {bench['sizeof_state_bytes']} B in report",
        )
        self.check(
            f"{bench['sizeof_config_bytes']:,} B" in report_text,
            f"Config struct size grounded: {bench['sizeof_config_bytes']} B in report",
        )
        self.check(
            f"{bench['actor_params_bytes']:,} B" in report_text,
            f"Actor weights footprint grounded: {bench['actor_params_bytes']} B in report",
        )

        # Latency claims
        mean_us_str = f"{bench['mean_us']:.3f}"
        self.check(
            mean_us_str in report_text,
            f"Benchmark mean latency grounded: {mean_us_str} us in report",
        )
        self.check(
            bench["mean_us"] < 10.0,
            f"Host mean execution latency {bench['mean_us']:.3f} us < 10.0 us budget",
        )
        self.check(
            bench["max_us"] < 100.0,
            f"Host max execution latency {bench['max_us']:.3f} us < 100.0 us budget",
        )

        # Equivalence claims
        self.check(equiv["mode_mismatches"] == 0, "Equivalence: exactly 0 mode mismatches")
        self.check(equiv["flags_mismatches"] == 0, "Equivalence: exactly 0 flags mismatches")
        self.check(
            "0 mismatches (100.000% match)" in report_text,
            "100.000% bit-exact mode match verified in report",
        )

    def audit_phase7_claims(self) -> None:
        print("\n--- Auditing Phase 7 Claims (Pre-Registered Test Evaluation) ---")
        stats_path = RESULTS_DIR / "phase7" / "stats.json"
        report_path = DOCS_DIR / "phase7_report.md"
        model_card_path = DOCS_DIR / "model_card_b5.md"

        self.check(stats_path.exists(), "results/phase7/stats.json exists")
        self.check(report_path.exists(), "docs/phase7_report.md exists")
        self.check(model_card_path.exists(), "docs/model_card_b5.md exists")

        if not (stats_path.exists() and report_path.exists()):
            return

        with open(stats_path, encoding="utf-8") as f:
            stats = json.load(f)
        report_text = report_path.read_text(encoding="utf-8")
        mc_text = model_card_path.read_text(encoding="utf-8")

        # Hypotheses
        h1 = stats["hypotheses_verdict"]["H1"]
        h2 = stats["hypotheses_verdict"]["H2"]
        h3 = stats["hypotheses_verdict"]["H3"]

        self.check(h1["pass"] and h1["b5_diverged"] == 0, "H1 passed: B5 0 test divergences")
        self.check(h2["pass"] and h2["b3_diverged"] == 31, "H2 passed: B3 31 test divergences")
        self.check(
            h3["pass"] and h3["b5_p95_max_cm"] < 50.0,
            f"H3 passed: B5 p95 max error {h3['b5_p95_max_cm']:.2f} cm < 50.0 cm",
        )

        # Cross-reference with report and model card
        self.check(
            "| b5_rl_supervised | 48 | 0 |" in report_text and "0 / 48 (0.0%)" in mc_text,
            "B5 0/48 test divergence claim grounded across report and model card",
        )
        self.check(
            "| b3_rl_unsupervised | 48 | 31 |" in report_text and "31 / 48 (64.6%)" in mc_text,
            "B3 31/48 test divergence claim grounded across report and model card",
        )
        self.check(
            "1.447 cm" in report_text and "1.45 cm" in mc_text,
            "B5 p95 lateral error (1.447 cm / 1.45 cm) grounded across report and model card",
        )

    def audit_phase6_claims(self) -> None:
        print("\n--- Auditing Phase 6 Claims (PPO Training & Validation Campaign) ---")
        stats_path = RESULTS_DIR / "phase6" / "stats.json"
        train_csv_path = RESULTS_DIR / "phase6" / "train_runs.csv"
        val_csv_path = RESULTS_DIR / "phase6" / "val_runs.csv"
        report_path = DOCS_DIR / "phase6_report.md"
        mc_path = DOCS_DIR / "model_card_b5.md"

        self.check(stats_path.exists(), "results/phase6/stats.json exists")
        self.check(train_csv_path.exists(), "results/phase6/train_runs.csv exists")
        self.check(val_csv_path.exists(), "results/phase6/val_runs.csv exists")
        self.check(report_path.exists(), "docs/phase6_report.md exists")
        self.check(mc_path.exists(), "docs/model_card_b5.md exists")

        if not (
            stats_path.exists()
            and train_csv_path.exists()
            and val_csv_path.exists()
            and report_path.exists()
            and mc_path.exists()
        ):
            return

        with open(stats_path, encoding="utf-8") as f:
            stats = json.load(f)
        report_text = report_path.read_text(encoding="utf-8")
        mc_text = mc_path.read_text(encoding="utf-8")

        def count_divergences(csv_path: Path, controller_name: str) -> int:
            count = 0
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    is_match = row.get("controller") == controller_name
                    is_div = row.get("diverged", "").strip().lower() in ("true", "1")
                    if is_match and is_div:
                        count += 1
            return count

        b5_train_div = count_divergences(train_csv_path, "b5_rl_supervised")
        b5_val_div = count_divergences(val_csv_path, "b5_rl_supervised")
        b3_train_div = count_divergences(train_csv_path, "b3_rl_unsupervised")
        b3_val_div = count_divergences(val_csv_path, "b3_rl_unsupervised")

        self.check(b5_train_div == 0, f"B5 train divergences = {b5_train_div} (expected 0)")
        self.check(b5_val_div == 0, f"B5 val divergences = {b5_val_div} (expected 0)")
        self.check(b3_train_div == 3, f"B3 train divergences = {b3_train_div} (expected 3)")
        self.check(b3_val_div == 4, f"B3 val divergences = {b3_val_div} (expected 4)")

        self.check(stats["train"]["n_scenarios"] == 48, "Phase 6 train scenarios = 48")
        self.check(stats["val"]["n_scenarios"] == 24, "Phase 6 val scenarios = 24")

        self.check("0 / 48" in mc_text, "B5 0/48 train grounded in model card")
        self.check("0 / 24" in mc_text, "B5 0/24 val grounded in model card")
        self.check("3 / 48" in mc_text, "B3 3/48 train grounded in model card")
        self.check("4 / 24" in mc_text, "B3 4/24 val grounded in model card")
        self.check(
            "| b5_rl_supervised | 48 | 0 |" in report_text,
            "B5 0/48 train grounded in phase6 report",
        )
        self.check(
            "| b5_rl_supervised | 24 | 0 |" in report_text,
            "B5 0/24 val grounded in phase6 report",
        )

    def run(self) -> int:
        print("=================================================================")
        print("                 EGGA CLAIM & EVIDENCE AUDITOR                  ")
        print("=================================================================")

        self.audit_traceability()
        self.audit_sotif_triggers()
        self.audit_phase6_claims()
        self.audit_phase7_claims()
        self.audit_phase8_claims()

        print("\n=================================================================")
        print(f"Audit Summary: {self.passed_checks} / {self.total_checks} checks passed.")
        if self.failures:
            print(f"FAILED CHECKS ({len(self.failures)}):")
            for f in self.failures:
                print(f"  - {f}")
            print("STATUS: AUDIT REJECTED (Discrepancies found)")
            print("=================================================================")
            return 1

        print("STATUS: AUDIT CERTIFIED (100% evidence-grounded, zero discrepancies)")
        print("=================================================================")
        return 0


if __name__ == "__main__":
    auditor = ClaimAuditor()
    sys.exit(auditor.run())
