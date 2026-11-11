"""Fail-closed lock on the sharp elliptical-hole tip (``K_t = 7``) verification.

The debt this locks is the one left open by ``docs/elliptic-hole-inglis-validation.md``:
an honest FAIL of the 3 % peak gate (8.872 %) at ``n = 48`` on the published
``grading = 3`` ladder.  ``docs/elliptic-hole-sharp-tip-certification.md`` shows the
lever is the *radial grading strength*, not ``n``; these tests keep both halves of
that statement true:

* the certifying configuration stays inside the gate,
* the legacy configuration stays *outside* it (so no one can claim the old run
  passed), and
* the tip-resolution diagnostic that predicts both keeps its analytic values.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from tensorfem.elliptic_hole import InglisHole, solve_case, tip_resolution

SHARP = InglisHole(semi_x=0.5, semi_y=1.5)       # K_t = 7, rho_t = 1/6
MODERATE = InglisHole(semi_x=1.5, semi_y=1.0)    # K_t = 2.333, rho_t = 2.25
EVIDENCE = pathlib.Path(__file__).resolve().parents[1] / "benchmarks" / "elliptic_hole_sharp_tip.json"

_CACHE: dict[tuple, dict] = {}


def case(hole: InglisHole, level: int, grading: float) -> dict:
    key = (hole.semi_x, hole.semi_y, level, grading)
    if key not in _CACHE:
        _CACHE[key] = solve_case(hole, n_theta=level, n_radial=level, grading=grading)
    return _CACHE[key]


# ---------------------------------------------------------------------------
# the certifying side of the debt
# ---------------------------------------------------------------------------

def test_certifying_configuration_inside_gate():
    """n=48, grading=6: peak 6.98464 vs K_t=7 -> 0.219 %, inside the 3 % gate."""
    solved = case(SHARP, 48, 6.0)
    error = solved["transverse_tip"]["relative_error"]
    assert error < 0.03, f"peak error {100 * error:.3f}% left the 3% gate"
    assert error < 0.01, f"peak error {100 * error:.3f}% lost its measured margin"
    assert solved["transverse_tip"]["exact"] == pytest.approx(7.0)
    # the same solve is still a *balanced* mesh: the element at the tip is not flat
    assert 0.5 < solved["tip_resolution"]["aspect"] < 3.0
    # global sanity checks stay at solver round-off
    assert solved["force_balance_relative"] < 1e-8
    assert solved["displacement_error"]["max_relative"] < 1e-3


def test_certifying_configuration_is_insensitive_to_level():
    """grading=6 is inside the gate at every affordable level (no lucky point)."""
    for level in (24, 32):
        solved = case(SHARP, level, 6.0)
        assert solved["transverse_tip"]["relative_error"] < 0.03, (
            f"n={level}, grading=6: {100 * solved['transverse_tip']['relative_error']:.3f}%"
        )
        assert solved["tip_resolution"]["radial_over_curvature"] < 0.01


# ---------------------------------------------------------------------------
# the legacy side of the debt: must stay an honest FAIL
# ---------------------------------------------------------------------------

def test_legacy_grading_family_still_fails_the_gate():
    """grading=3 was and stays a FAIL; the published 8.872 % run is not upgraded."""
    legacy = case(SHARP, 32, 3.0)
    assert legacy["transverse_tip"]["relative_error"] > 0.03, (
        "grading=3 now passes -- if the production path changed, re-run the audit "
        "and update the report instead of silently dropping the honest FAIL"
    )
    # and the reason is visible in the diagnostic, at every level of that family
    for level in (24, 32, 48):
        res = tip_resolution(SHARP, level, level, 3.0)
        assert res["radial_over_curvature"] >= 0.03


def test_load_axis_tip_control_is_accurate_everywhere():
    """Control: the load-axis tip has no curvature concentration -> always inside."""
    for grading, level in ((3.0, 24), (3.0, 32)):
        solved = case(SHARP, level, grading)
        assert solved["load_axis_tip"]["relative_error"] < 0.02
        assert solved["load_axis_tip"]["exact"] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# the diagnostic itself, against hand-computed analytic values
# ---------------------------------------------------------------------------

def test_tip_resolution_analytic_values():
    """|dz/dzeta| = R(1+m) at the tip; grid spacings are public formulas."""
    metric = SHARP.mapping_radius * (1.0 + SHARP.mapping_m)   # 1.0 * 0.5 = 0.5
    res = tip_resolution(SHARP, 48, 48, 6.0)
    assert res["curvature_radius"] == pytest.approx(SHARP.semi_x**2 / SHARP.semi_y)
    assert res["radial_layer"] == pytest.approx(0.0006617, rel=1e-3)
    assert res["angular_layer"] == pytest.approx(0.0008411, rel=1e-3)
    assert res["aspect"] == pytest.approx(res["angular_layer"] / res["radial_layer"])
    assert res["radial_over_curvature"] == pytest.approx(res["radial_layer"] / (1.0 / 6.0))
    # grading=0 is a uniform ring: spacing = 4/n_radial * metric
    uniform = tip_resolution(SHARP, 42, 42, 0.0)
    assert uniform["radial_layer"] == pytest.approx(4.0 / 42.0 * metric)
    # a stronger grading makes the tip cells flatter radially, i.e. aspect grows
    assert tip_resolution(SHARP, 32, 32, 3.0)["aspect"] < tip_resolution(SHARP, 32, 32, 6.0)["aspect"]


def test_moderate_hole_predicts_its_own_pass():
    """The same diagnostic explains why case A passed at grading=3 (rho_t = 2.25)."""
    res = tip_resolution(MODERATE, 48, 48, 3.0)
    assert res["radial_over_curvature"] < 0.01
    solved = case(MODERATE, 32, 3.0)
    assert solved["transverse_tip"]["relative_error"] < 0.05


# ---------------------------------------------------------------------------
# recorded evidence artifact
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not EVIDENCE.exists(), reason="run scripts/audit_elliptic_hole_sharp_tip.py")
def test_recorded_evidence_partition_is_consistent():
    report = json.loads(EVIDENCE.read_text())
    resolved = report["rule"]["resolved_condition"]
    unresolved = report["rule"]["unresolved_condition"]
    assert resolved["cases"] >= 10 and resolved["all_within_gate"] is True
    assert all(row["peak_error_pct"] < report["gate_pct"] for row in resolved["cases_table"])
    assert unresolved["cases"] >= 3 and unresolved["all_fail_gate"] is True
    assert all(row["peak_error_pct"] > report["gate_pct"] for row in unresolved["cases_table"])
    assert report["certified"]["peak_error_pct"] < report["gate_pct"]
    assert report["legacy"]["exceeds_gate"] is True
    # the old family's fitted order is ~1, so 3 % would need a far larger ladder
    family = report["families"]["B_transverse_sharp"]["grading_3"]
    assert 0.8 < family["observed_order"] < 1.1
    assert family["level_for_3pct"] > 100