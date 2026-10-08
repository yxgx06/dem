from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from egga.config import REPO_ROOT, load_config
from egga.eval.baselines import run_scenario
from egga.scenarios import sets
from egga.scenarios.sets import generate_set, load_scenarios, run_arguments, set_hash

NAMES = ("train", "val", "test")


@pytest.mark.parametrize("name", NAMES)
def test_regenerated_set_matches_frozen_hash(name: str) -> None:
    manifest = json.loads((sets.SET_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert set_hash(generate_set(name)) == manifest["hashes"][name]


def test_generation_is_deterministic() -> None:
    assert generate_set("val") == generate_set("val")


def test_train_and_val_load_with_hash_check() -> None:
    assert len(load_scenarios("train")) == 48
    assert len(load_scenarios("val")) == 24


def test_test_set_is_locked_without_explicit_unlock() -> None:
    with pytest.raises(PermissionError):
        load_scenarios("test")
    assert len(load_scenarios("test", unlock_test=True)) == 48


def test_no_source_file_unlocks_the_test_set_before_phase7() -> None:
    offenders = []
    for folder in ("src", "tools"):
        for path in (REPO_ROOT / folder).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "unlock_test=True" in text and path.name not in ("sets.py", "phase7.py"):
                offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == [], f"test set unlocked outside Phase 7: {offenders}"


def test_tampered_set_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    copy = tmp_path / "scenario_sets"
    shutil.copytree(sets.SET_DIR, copy)
    payload = json.loads((copy / "val.json").read_text(encoding="utf-8"))
    payload["scenarios"][0]["speed_mps"] += 0.5
    (copy / "val.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(sets, "SET_DIR", copy)
    with pytest.raises(ValueError):
        load_scenarios("val")


def test_slice_indices_are_disjoint_across_sets() -> None:
    cfg = load_config("scenario_sets.yaml")
    indices = [set(cfg["sets"][n]["slice_indices"]) for n in NAMES]
    assert indices[0].isdisjoint(indices[1])
    assert indices[0].isdisjoint(indices[2])
    assert indices[1].isdisjoint(indices[2])
    assert all(i < cfg["slices"] for s in indices for i in s)


def test_every_parameter_value_lies_in_its_sets_slices_only() -> None:
    cfg = load_config("scenario_sets.yaml")
    slices = cfg["slices"]
    for name in NAMES:
        allowed = set(cfg["sets"][name]["slice_indices"])
        for spec in generate_set(name):
            for key, index in spec["slice_indices"].items():
                assert index in allowed
                lo, hi = cfg["ranges"][key]
                value = {
                    "speed_mps": spec["speed_mps"],
                    "mass_scale": spec["mass_scale"],
                    "delay_s": spec["delay_s"],
                    "noise_std_m": spec["noise_std_m"],
                    "mu_belief_error": spec["mu_belief_error"],
                    "demand_ratio": spec["demand_ratio"],
                }.get(key)
                if value is not None:
                    assert int((value - lo) / (hi - lo) * slices) == index or value == hi


def test_seeds_and_ids_are_unique_and_blocks_disjoint() -> None:
    seeds = {n: {s["seed"] for s in generate_set(n)} for n in NAMES}
    assert seeds["train"].isdisjoint(seeds["val"]) and seeds["train"].isdisjoint(seeds["test"])
    assert seeds["val"].isdisjoint(seeds["test"])
    ids = [s["id"] for n in NAMES for s in generate_set(n)]
    assert len(ids) == len(set(ids))


def test_scenarios_are_feasible_by_construction_and_cover_all_families() -> None:
    cap = load_config("scenario_sets.yaml")["geometry"]["max_lc_amp_rad"]
    for name in NAMES:
        specs = generate_set(name)
        assert all(s["demand_ratio"] <= 1.0 for s in specs)
        assert all(0.0 < s["geometry"]["lc_amp_rad"] <= cap for s in specs)
        families = set(load_config("scenario_sets.yaml")["families"])
        assert {s["friction_family"] for s in specs} == families


def test_train_covers_near_limit_demand() -> None:
    assert max(s["demand_ratio"] for s in generate_set("train")) > 0.75


def test_run_arguments_build_a_valid_flat_mission() -> None:
    spec = load_scenarios("val")[0]
    plant, mission, belief, seed = run_arguments(spec)
    assert mission["t_final_s"] == spec["duration_s"] and seed == spec["seed"]
    assert plant["friction"]["profile"] == spec["friction"]
    assert abs(belief) <= 0.2


def test_scenario_run_is_deterministic_and_bounded_for_pd_ff() -> None:
    spec = load_scenarios("val")[0]
    a = run_scenario("b1_pd_ff", spec)
    b = run_scenario("b1_pd_ff", spec)
    np.testing.assert_array_equal(a.ey, b.ey)
    assert a.t[-1] == pytest.approx(spec["duration_s"])
