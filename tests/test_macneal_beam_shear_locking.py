"""CI gates for the MacNeal (1985) slender-beam shear-locking anchor.

The anchor (``benchmarks/macneal_beam_shear_locking/``) shipped with an
ad-hoc ``validate.py`` but no test in the repository suite, so nothing stopped
the anchor's numbers or its ``shear_treatment="full"`` option from silently
drifting.  These tests re-derive every graded quantity from the code:

* the two closed-form references are recomputed here from ``P, L, E, I, k, A,
  G`` (never read from the evidence JSON);
* the locking-free rules must meet the 5 % anchor tolerance at ``n = 16`` and
  the equal-order rule must miss it (the locking signature);
* ``results.json`` must reproduce from a fresh solve;
* the pre-registered asymptotic-rate study
  (``locking_rate_study.py``, ``locking_rate.json``) is pinned: its thresholds,
  its arithmetic and its decision must agree with the stored numbers.

Ladder sizes are chosen so the whole module stays in the fast lane (a few
seconds); the full ``n = 2..256`` reproduction lives in the study script.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ANCHOR = ROOT / "benchmarks" / "macneal_beam_shear_locking"

# Case constants of MacNeal (1985): L=10, h=1, b=1, E=1, nu=0.3, P=1, k=5/6.
E, NU, L, H, B, P, K = 1.0, 0.3, 10.0, 1.0, 1.0, 1.0, 5.0 / 6.0


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUN = _load("macneal_anchor_run", ANCHOR / "run.py")
STUDY = _load("macneal_anchor_rate_study", ANCHOR / "locking_rate_study.py")


def _closed_form():
    """Independent Euler-Bernoulli / Timoshenko centre deflections."""
    inertia = B * H**3 / 12.0
    area = B * H
    shear_modulus = E / (2.0 * (1.0 + NU))
    eb = P * L**3 / (48.0 * E * inertia)
    shear = P * L / (4.0 * K * area * shear_modulus)
    return eb, eb + shear, shear


# --------------------------------------------------------------------------- #
# References: closed form, re-derived here
# --------------------------------------------------------------------------- #
def test_reference_closed_forms_are_euler_bernoulli_and_timoshenko():
    eb, timoshenko, shear = _closed_form()
    assert eb == pytest.approx(250.0, rel=1e-14)
    assert shear == pytest.approx(7.8, rel=1e-12)  # PL/(4kAG), k = 5/6
    assert timoshenko == pytest.approx(257.8, rel=1e-14)

    analytic = RUN.beam_analytic()
    assert analytic["euler_bernoulli"] == pytest.approx(eb, rel=1e-14)
    assert analytic["timoshenko"] == pytest.approx(timoshenko, rel=1e-14)
    assert analytic["inertia"] == pytest.approx(B * H**3 / 12.0, rel=1e-15)
    assert analytic["slenderness"] == pytest.approx(L / H, rel=1e-15)
    assert analytic["shear_correction_pct"] == pytest.approx(100 * 7.8 / 250.0, rel=1e-12)


def test_results_json_references_match_the_closed_form():
    stored = json.loads((ANCHOR / "results.json").read_text())["analytic"]
    eb, timoshenko, _ = _closed_form()
    assert stored["euler_bernoulli"] == pytest.approx(eb, rel=1e-12)
    assert stored["timoshenko"] == pytest.approx(timoshenko, rel=1e-12)


# --------------------------------------------------------------------------- #
# Element behaviour at the anchor mesh
# --------------------------------------------------------------------------- #
def test_locking_free_rules_meet_the_anchor_tolerance():
    eb, timoshenko, _ = _closed_form()
    for treatment in ("sri", "mitc"):
        delta = RUN.solve_beam(16, treatment)["centre_deflection"]
        assert abs(delta - timoshenko) / timoshenko < 0.03  # < 3 %: task bar
        assert abs(delta - eb) / eb < 0.05  # < 5 %: anchor gate


def test_equal_order_rule_locks_at_the_anchor_mesh():
    _, timoshenko, _ = _closed_form()
    full = RUN.solve_beam(16, "full")["centre_deflection"]
    free = RUN.solve_beam(16, "sri")["centre_deflection"]
    err_full = abs(full - timoshenko) / timoshenko
    err_free = abs(free - timoshenko) / timoshenko
    assert err_full > 0.05  # the documented locking failure
    assert full < timoshenko  # locking stiffens: deflection is low
    assert err_full > 10.0 * err_free  # and it is the least accurate rule


def test_strip_mesh_and_boundary_conditions():
    nodes, elements = RUN.build_strip(4)
    assert nodes.shape == (10, 2)
    assert elements.shape == (4, 4)
    assert float(nodes[:, 0].max()) == pytest.approx(L, abs=1e-12)
    assert sorted(set(nodes[:, 1].tolist())) == [0.0, 1.0]
    # the mid-span node exists at x = L/2 (all anchor levels are even)
    assert ((nodes[:, 0] - L / 2).abs() < 1e-12).any()
    # SRI and MITC coincide on this rectangular single-element-wide strip
    sri = RUN.solve_beam(8, "sri")["centre_deflection"]
    mitc = RUN.solve_beam(8, "mitc")["centre_deflection"]
    assert sri == pytest.approx(mitc, rel=1e-9)


def test_unknown_shear_treatment_is_rejected():
    nodes, _ = RUN.build_strip(2)
    with pytest.raises(ValueError):
        RUN.q4_mindlin_stiffness(
            nodes[:4], shear_treatment="not-a-rule", rigidity=RUN.beam_rigidity()
        )


# --------------------------------------------------------------------------- #
# results.json must reproduce from a fresh solve
# --------------------------------------------------------------------------- #
def test_results_json_rows_reproduce():
    payload = json.loads((ANCHOR / "results.json").read_text())
    rows = {(r["treatment"], r["n_elements"]): r for r in payload["rows"]}
    for treatment in payload["treatments"]:
        stored = rows[(treatment, 4)]["centre_deflection"]
        fresh = RUN.solve_beam(4, treatment)["centre_deflection"]
        assert stored == pytest.approx(fresh, rel=1e-10)
    assert payload["levels"] == [2, 4, 8, 16]
    assert set(payload["treatments"]) == {"full", "sri", "mitc"}


# --------------------------------------------------------------------------- #
# Pre-registered rate study
# --------------------------------------------------------------------------- #
def test_local_rate_machinery():
    levels = [4, 8, 16]
    rates = STUDY.local_rates(levels, [0.060609, 0.015152, 0.003788])
    assert [r["n_from"] for r in rates] == [4, 8]
    for r in rates:
        assert r["rate"] == pytest.approx(2.0, abs=0.01)
    with pytest.raises(ValueError):
        STUDY.local_rates([2, 4], [0.0, 0.1])
    with pytest.raises(ValueError):
        STUDY.asymptotic_rate([2, 4], [0.1], count=3)


def test_mesh_for_tolerance_interpolates_and_reports_misses():
    levels = [2, 4, 8, 16]
    errors = [-0.242436, -0.060609, -0.015152, -0.003788]
    n_tol = STUDY.mesh_for_tolerance(levels, errors, 0.05)
    assert 4.0 < n_tol < 8.0
    # log-log interpolation between the n=4 and n=8 rungs
    assert n_tol == pytest.approx(4.0 * 2 ** (math.log(0.060609 / 0.05) / math.log(4.0)), rel=1e-4)
    assert STUDY.mesh_for_tolerance(levels, [-0.01] * 4, 0.05) == 2.0  # floor
    assert STUDY.mesh_for_tolerance(levels, [-1.0] * 4, 0.05) is None  # never reached


def test_preasymptotic_rate_rises_to_second_order():
    """Cheap ladder: the equal-order rule starts below O(h^2) and climbs to it."""
    eb, timoshenko, _ = _closed_form()
    levels = (2, 4, 8, 16, 32)
    errors = {}
    for treatment in ("full", "sri"):
        errors[treatment] = [
            (RUN.solve_beam(n, treatment)["centre_deflection"] - timoshenko) / timoshenko
            for n in levels
        ]
    free_rates = [r["rate"] for r in STUDY.local_rates(levels, errors["sri"])]
    for rate in free_rates:
        assert rate == pytest.approx(2.0, abs=1e-3)  # exact O(h^2)
    full_rates = [r["rate"] for r in STUDY.local_rates(levels, errors["full"])]
    assert full_rates[0] < 1.0  # strongly pre-asymptotic
    assert full_rates == sorted(full_rates)  # monotonically climbing
    assert full_rates[-1] > 1.8  # ... towards O(h^2)
    assert full_rates[2] < 1.8  # anchor mesh is not settled


def test_rate_study_json_is_reproducible_and_decision_consistent():
    payload = json.loads((ANCHOR / "locking_rate.json").read_text())
    protocol = payload["protocol"]
    pre = protocol["pre_registered"]
    # thresholds must equal the module's pre-registered constants
    assert pre["rate_const_min"] == STUDY.RATE_CONST_MIN
    assert pre["rate_const_spread"] == STUDY.RATE_CONST_SPREAD
    assert pre["rate_defect_max"] == STUDY.RATE_DEFECT_MAX
    assert pre["mesh_penalty_min"] == STUDY.MESH_PENALTY_MIN
    assert protocol["levels"] == list(STUDY.LEVELS)

    levels = protocol["levels"]
    series = payload["series"]
    # every rate / mesh figure in the artifact is re-derived from its own errors
    for treatment in protocol["treatments"]:
        errors = series[treatment]["errors"]
        assert series[treatment]["asymptotic_rate"] == pytest.approx(
            STUDY.asymptotic_rate(levels, errors), rel=1e-10
        )
        assert series[treatment]["mesh_for_5pct"] == pytest.approx(
            STUDY.mesh_for_tolerance(levels, errors), rel=1e-10
        )
        for stored, fresh in zip(
            series[treatment]["local_rates"], STUDY.local_rates(levels, errors)
        ):
            assert stored["rate"] == pytest.approx(fresh["rate"], rel=1e-10)

    # the pre-registered decision and its gate block
    gates = payload["gates"]
    assert gates["H_const_asymptotic_rate"]["pass"] is True
    assert gates["H_rate_asymptotic_rate"]["pass"] is False
    assert gates["mesh_penalty_material"]["pass"] is True
    assert gates["anchor_mesh_is_preasymptotic"]["pass"] is True
    assert payload["decision"]["hypothesis"] == "H_const"
    assert series["full"]["asymptotic_rate"] >= STUDY.RATE_CONST_MIN
    assert (
        abs(series["full"]["asymptotic_rate"] - series["sri"]["asymptotic_rate"])
        <= STUDY.RATE_CONST_SPREAD
    )
    # the error-constant ratio predicts the measured mesh penalty to a few %
    ratio = abs(series["full"]["errors"][-1]) / abs(series["sri"]["errors"][-1])
    assert ratio == pytest.approx(
        payload["locking"]["error_constant_ratio_full_over_sri"], rel=1e-10
    )
    penalty = series["full"]["mesh_for_5pct"] / series["sri"]["mesh_for_5pct"]
    assert penalty == pytest.approx(math.sqrt(ratio), rel=0.10)
