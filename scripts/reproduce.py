from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from egga.config import REPO_ROOT


class ReproductionPipeline:
    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self.passed_steps: list[str] = []
        self.failed_steps: list[str] = []
        self.durations: dict[str, float] = {}

    def run_step(self, name: str, cmd: list[str], cwd: Path | None = None) -> bool:
        print(f"\n>>> [RUNNING] {name}")
        print(f"    Command: {' '.join(cmd)}")
        start_time = time.perf_counter()

        try:
            res = subprocess.run(
                cmd,
                cwd=cwd or REPO_ROOT,
                check=False,
                text=True,
                capture_output=not self.verbose,
            )
            duration = time.perf_counter() - start_time
            self.durations[name] = duration

            if res.returncode == 0:
                print(f"    [PASSED] {name} ({duration:.2f}s)")
                self.passed_steps.append(name)
                return True
            else:
                print(f"    [FAILED] {name} (exit code: {res.returncode}, {duration:.2f}s)")
                if not self.verbose and res.stderr:
                    print(res.stderr)
                self.failed_steps.append(name)
                return False
        except Exception as e:
            duration = time.perf_counter() - start_time
            self.durations[name] = duration
            print(f"    [ERROR] {name}: {e}")
            self.failed_steps.append(name)
            return False

    def print_summary(self) -> int:
        total = len(self.passed_steps) + len(self.failed_steps)
        total_time = sum(self.durations.values())

        print("\n" + "=" * 70)
        print("                 EGGA MASTER REPRODUCTION REPORT")
        print("=" * 70)
        for step_name in self.passed_steps:
            dur = self.durations.get(step_name, 0.0)
            print(f"  [PASS] {step_name:<50} ({dur:6.2f}s)")
        for step_name in self.failed_steps:
            dur = self.durations.get(step_name, 0.0)
            print(f"  [FAIL] {step_name:<50} ({dur:6.2f}s)")

        print("-" * 70)
        print(f"Total Steps: {len(self.passed_steps)} / {total} Passed in {total_time:.2f}s")
        if self.failed_steps:
            print("STATUS: REPRODUCTION FAILED")
            print("=" * 70)
            return 1
        print("STATUS: REPRODUCTION VERIFIED & CERTIFIED")
        print("=" * 70)
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="EGGA Master End-to-End Reproduction & Verification Pipeline"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run fast verification (lint, type-check, C99 build, equivalence, claim auditor)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run full verification (including 100%% supervisor coverage and full benchmark)",
    )
    parser.add_argument(
        "--c99-only",
        action="store_true",
        help="Run only C99 compilation, equivalence, and benchmark",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Run only the formal claim and evidence auditor",
    )
    parser.add_argument(
        "--cov-only",
        action="store_true",
        help="Run only supervisor 100%% branch coverage pytest suite",
    )
    args = parser.parse_args()

    py = sys.executable
    pipeline = ReproductionPipeline(verbose=True)

    if args.audit_only:
        pipeline.run_step("Claim & Evidence Auditor", [py, "tools/claim_auditor.py"])
        return pipeline.print_summary()

    if args.cov_only:
        pipeline.run_step(
            "Supervisor 100% Branch Coverage",
            [
                py,
                "-m",
                "pytest",
                "--cov=src/egga/supervisor",
                "--cov-report=term-missing",
                "--cov-fail-under=100",
                "tests/test_supervisor.py",
                "tests/test_envelope.py",
                "tests/test_supervisor_properties.py",
                "tests/redteam/test_supervisor_redteam.py",
            ],
        )
        return pipeline.print_summary()

    if args.c99_only:
        pipeline.run_step(
            "Build & Verify C99 Library",
            [py, "-c", "from egga.c_bridge import build_c_lib; build_c_lib()"],
        )
        pipeline.run_step(
            "C99 Bit-Exact Equivalence",
            [py, "-m", "pytest", "tests/test_c99_equivalence.py"],
        )
        pipeline.run_step(
            "C99 Embedded Benchmark",
            [py, "tools/benchmark_c99.py", "--steps", "10000"],
        )
        return pipeline.print_summary()

    # Step 1: Linting
    pipeline.run_step(
        "Code Linting (Ruff)",
        [py, "-m", "ruff", "check", "src", "tests", "tools", "scripts"],
    )

    # Step 2: Strict Type Checking
    pipeline.run_step(
        "Strict Static Typing (Mypy)",
        [
            py,
            "-m",
            "mypy",
            "--strict",
            "src/egga/supervisor/",
            "src/egga/estimation/",
            "tools/claim_auditor.py",
            "scripts/reproduce.py",
        ],
    )

    # Step 3: C99 Build & Equivalence
    pipeline.run_step(
        "Compile C99 Embedded Engine",
        [py, "-c", "from egga.c_bridge import build_c_lib; build_c_lib()"],
    )
    pipeline.run_step(
        "C99 / Python Equivalence Tests",
        [py, "-m", "pytest", "tests/test_c99_equivalence.py"],
    )

    # Step 4: Supervisor Coverage (or fast check)
    if args.fast:
        pipeline.run_step(
            "Supervisor Core Unit Tests",
            [py, "-m", "pytest", "tests/test_supervisor.py"],
        )
    else:
        pipeline.run_step(
            "Supervisor 100% Branch Coverage Suite",
            [
                py,
                "-m",
                "pytest",
                "--cov=src/egga/supervisor",
                "--cov-report=term-missing",
                "--cov-fail-under=100",
                "tests/test_supervisor.py",
                "tests/test_envelope.py",
                "tests/test_supervisor_properties.py",
                "tests/redteam/test_supervisor_redteam.py",
            ],
        )

    # Step 5: Formal Claim Auditor
    pipeline.run_step("Claim & Evidence Auditor", [py, "tools/claim_auditor.py"])

    return pipeline.print_summary()


if __name__ == "__main__":
    sys.exit(main())
