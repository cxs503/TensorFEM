"""Contract tests for the ramp-density (load-step) refinement study.

Anchor ``GAP-PP-3-ICE-STRUCT-PLASTIC-SET``.  The study itself runs finite
elements on the queue; these tests pin down the *analysis* contract, i.e. the
part that decides what the measurements are allowed to claim:

* a step-refinement claim needs two **admissible** step sizes on the same mesh --
  otherwise the verdict is ``withheld`` rather than interpolated;
* a limit load is compared with the rigid-plastic bracket and being outside it
  is reported, not explained away;
* mesh effect is only ever read at a **matched** load step (that is the whole
  point: the archived ladder compared meshes run with different ramp density);
* a withheld (``None``) limit load stays withheld through the aggregation --
  no fail-open default.

The tests feed hand-built series/records (no FE), so they run in milliseconds
and do not depend on queue artifacts.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "patch_plate_plastic_step_refinement.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("stepref", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stepref = _load_module()


def record(nx, step, limit, *, admissible=True, peak=None, first=None,
           permanent=None, valid=False, bounds=None):
    """Minimal stand-in for one ramp artifact."""
    return {
        "tag": stepref.case_tag(nx, step),
        "nx": nx,
        "step_p_y": step,
        "limit_tight": limit,
        "limit_loose": limit,
        "admissible": admissible and limit is not None,
        "criteria_spread_tight": 0.01 if limit is not None else None,
        "peak_factor": peak if peak is not None else (limit or 1.0),
        "ramp_truncated": not admissible,
        "first_yield": first,
        "permanent_deflection": permanent,
        "permanent_set_valid": valid,
        "n_committed_points": 30,
        "elapsed_s": 1.0,
        "references": {"p_yield_analytic": 0.8692e6,
                       "bounds": bounds if bounds is not None else
                       {"strip_lower": 0.8188e6, "johansen_twist": 2.3746e6,
                        "yield_line": 1.8422e6, "ilyushin_sine": 1.5152e6}},
    }


# ---------------------------------------------------------------------------
# step-refinement verdicts
# ---------------------------------------------------------------------------

def test_two_admissible_steps_that_agree_are_step_independent():
    series = [record(12, 0.05, 1.700e6), record(12, 0.025, 1.701e6)]
    out = stepref.analyse_series(series)
    assert out["verdict"] == "step_independent"
    assert out["finest_two"]["relative_change"] == pytest.approx(1.701 / 1.700 - 1.0)


def test_two_steps_that_disagree_are_not_called_converged():
    series = [record(12, 0.05, 1.70e6), record(12, 0.025, 1.50e6)]
    out = stepref.analyse_series(series)
    assert out["verdict"] in {"converging", "not_converged"}
    assert out["verdict"] != "step_independent"


def test_single_admissible_step_withholds_the_claim():
    series = [record(12, 0.05, 1.70e6), record(12, 0.025, None, admissible=False)]
    out = stepref.analyse_series(series)
    assert out["verdict"] == "withheld"
    assert "reason" in out
    assert out["n_usable"] == 1


def test_withheld_limits_never_become_a_number():
    """Fail-closed: a path the gate rejected must not yield a limit load."""
    series = [record(16, 0.10, None, admissible=False),
              record(16, 0.05, None, admissible=False)]
    out = stepref.analyse_series(series)
    assert out["verdict"] == "withheld"
    assert all(v is None for v in out["limits"])


def test_richardson_order_is_used_when_three_steps_exist():
    # p(dp) = 1.4e6 + 3.0e5 * dp**1.5  ->  observed order should come back ~1.5
    series = [record(8, dp, 1.4e6 + 3.0e5 * dp ** 1.5)
              for dp in (0.10, 0.05, 0.025)]
    out = stepref.analyse_series(series)
    rich = out["step_richardson"]
    assert rich["order"] == pytest.approx(1.5, abs=0.15)
    assert rich["limit_dp_to_zero"] == pytest.approx(1.4e6, rel=0.02)
    assert out["verdict"] in {"converging", "step_independent"}


def test_non_monotone_differences_refuse_extrapolation():
    series = [record(8, 0.10, 1.00e6), record(8, 0.05, 1.30e6),
              record(8, 0.025, 1.31e6)]
    out = stepref.analyse_series(series)
    # differences shrink (300k -> 10k) => monotone; extrapolation allowed
    assert out["change_decreasing"] is True
    series_bad = [record(8, 0.10, 1.00e6), record(8, 0.05, 1.01e6),
                  record(8, 0.025, 1.80e6)]
    out_bad = stepref.analyse_series(series_bad)
    assert out_bad["change_decreasing"] is False
    assert out_bad["verdict"] == "not_converged"
    assert "step_richardson" not in out_bad


# ---------------------------------------------------------------------------
# mesh effect at matched ramp density
# ---------------------------------------------------------------------------

def test_mesh_effect_is_only_paired_at_equal_step():
    recs = [record(8, 0.05, 1.20e6), record(12, 0.05, 1.67e6),
            record(8, 0.10, 1.10e6), record(16, 0.10, 1.65e6)]
    out = stepref.matched_mesh_effect(recs)
    by_step = {entry["step_p_y"]: entry for entry in out}
    assert sorted(by_step) == [0.05, 0.10]
    # at dp = 0.05 the pair (8, 12) exists; the (12, 16) pair must not be
    # invented from mismatched ramps
    assert by_step[0.05]["meshes"] == [8, 12]
    assert by_step[0.10]["meshes"] == [8, 16]
    pair = by_step[0.05]["pairs"][0]
    assert pair["relative_change"] == pytest.approx(1.67 / 1.20 - 1.0)
    assert pair["same_ramp_density"] is True


def test_mesh_effect_reports_none_for_withheld_member():
    recs = [record(8, 0.05, 1.20e6), record(12, 0.05, None, admissible=False)]
    out = stepref.matched_mesh_effect(recs)
    assert out[0]["limits"] == [1.20e6, None]
    assert out[0]["pairs"][0]["relative_change"] is None


# ---------------------------------------------------------------------------
# bracket check
# ---------------------------------------------------------------------------

def test_bracket_check_withholds_without_limit():
    out = stepref.bracket_check(None, record(8, 0.05, None))
    assert out["verdict"] == "withheld"


def test_bracket_check_flags_a_value_outside_the_rigid_plastic_bracket():
    inside = stepref.bracket_check(1.60e6, record(8, 0.05, 1.60e6))
    assert inside["verdict"] == "inside"
    assert inside["inside_bracket"] is True
    outside = stepref.bracket_check(3.00e6, record(8, 0.05, 3.00e6))
    assert outside["verdict"] == "outside"
    assert outside["errors"]["johansen_twist"] > 0.0


def test_bracket_check_withholds_when_references_are_missing():
    rec = record(8, 0.05, 1.6e6)
    rec["references"] = {}
    out = stepref.bracket_check(1.6e6, rec)
    assert out["verdict"] == "withheld"


# ---------------------------------------------------------------------------
# the load-step series that is actually planned
# ---------------------------------------------------------------------------

def test_plan_covers_the_full_matched_step_grid():
    plan = stepref.PLAN
    assert len(plan) == len(set(plan)), "duplicate (mesh, step) case in PLAN"
    for nx in stepref.MESH_SEQUENCE:
        steps = sorted(dp for n, dp in plan if n == nx)
        assert steps == sorted(stepref.STEP_SEQUENCE), (
            f"mesh nx={nx} must have the full step series: {steps}")
    # the archive defect was comparing two meshes at different ramp densities,
    # so at least one step must be shared by all meshes
    common = set.intersection(*[{dp for n, dp in plan if n == m}
                                for m in stepref.MESH_SEQUENCE])
    assert common, "no load step shared by all meshes: mesh effect is confounded"


def test_case_tag_is_stable_and_filesystem_safe():
    assert stepref.case_tag(12, 0.05) == "nx12_dp0.05"
    assert "/" not in stepref.case_tag(8, 0.025)


def test_layered_plate_module_still_exposes_the_gate():
    """The study is only meaningful while the admissibility gate is enforced."""
    from tensorfem.plate_patch_plastic_limit import (
        limit_load_from_path, path_admissibility, run_patch_plastic_path)
    assert callable(path_admissibility) and callable(limit_load_from_path)
    assert callable(run_patch_plastic_path)
    assert torch is not None