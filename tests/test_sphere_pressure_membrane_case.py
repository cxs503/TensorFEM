"""CASE-SPHERE-PRESSURE gates: uniform-pressure sphere membrane stress ``pR/(2t)``.

Locks the case-pool anchor
``benchmarks/verified/fem_sphere_pressure_membrane`` into the test suite:

* the reference is *derived* (sympy equilibrium) and equals ``pR/(2t)`` with an
  exactly zero residual, so the value cannot drift by hand-editing;
* the production element (`AxisymShell`, circular-arc meridian) reproduces the
  membrane state inside the predeclared 3 % tolerance at an engineering mesh;
* both recorded traps stay traps (fail-closed): straight chord meridians and the
  flat-facet Shell4 path must miss by far more than 3 %;
* the committed evidence files agree with a fresh solve (no stale numbers).

Runtime: ~10 s (the flat-facet control carries the dense 24-facet solve), so the
whole module is marked ``slow`` only where it is genuinely heavy.
"""
from __future__ import annotations

import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorfem.axisym_shell_benchmarks import sphere_pressure  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CASE_DIR = os.path.join(ROOT, "benchmarks", "verified", "fem_sphere_pressure_membrane")
sys.path.insert(0, CASE_DIR)

CASE = {"radius": 10.0, "thickness": 0.04, "young": 6.825e7, "poisson": 0.3, "pressure": 1.0}
TOL = 0.03


# --------------------------------------------------------------------------- ref
def test_reference_is_derived_not_transcribed():
    import sympy as sp

    import run_verification as rv

    ref = rv.reference()
    # two independent statics arguments must cancel symbolically at r = R
    assert ref["derivation"]["residual_axial_vs_laplace_at_r_R"] == "0"
    assert ref["residual_is_zero"] is True

    # the stored expression must be symbolically p*R/(2t) - checked as algebra,
    # not as a string, so an edit of the closed form cannot slip through
    p, R, t, E, nu = sp.symbols("p R t E nu", positive=True)
    loc = {"p": p, "R": R, "t": t, "E": E, "nu": nu}
    sigma = sp.sympify(ref["derivation"]["sigma_expr"], locals=loc)
    w = sp.sympify(ref["derivation"]["w_expr"], locals=loc)
    assert sp.simplify(sigma - p * R / (2 * t)) == 0
    assert sp.simplify(w - p * R ** 2 * (1 - nu) / (2 * E * t)) == 0

    # The Lame mid-surface series in eps = t/R is p/(2 eps) - p/2 + eps p/12 + ...
    # Its divergent-looking leading term *is* the thin-shell law, p/(2 eps) = pR/(2t);
    # the O(1) offset -p/2 is what makes the thin-shell formula differ from 3-D
    # elasticity by a relative amount of order eps - the finding of section 5-A.
    eps = sp.Symbol("eps", positive=True)
    series = sp.sympify(ref["derivation"]["lame_mid_surface_series_t_over_R"],
                        locals={**loc, "eps": eps})
    assert sp.simplify(series - (p / (2 * eps) - p / 2 + eps * p / 12)) == 0
    assert sp.simplify(series.subs(eps, t / R) - p * R / (2 * t)) != 0  # the O(1) offset

    values = ref["values"]
    assert values["sigma_pa"] == pytest.approx(125.0, rel=1e-14)
    assert values["w_m"] == pytest.approx(1.2820512820512818e-05, rel=1e-14)
    assert ref["provenance"]["page_level_status"] == "WITHHELD"


def test_reference_matches_3d_elasticity_limit():
    """Lame thick-sphere mid-surface hoop stress -> pR/(2t) as t/R -> 0."""
    import torch

    from tensorfem.sphere_pressure_benchmark import lame_reference

    R, E, nu, p = CASE["radius"], CASE["young"], CASE["poisson"], CASE["pressure"]
    devs = []
    for ratio in (1e-3, 1e-3 / 4):
        t = ratio * R
        prob = {"inner_radius_m": R - t / 2, "outer_radius_m": R + t / 2,
                "young_modulus_pa": E, "poisson_ratio": nu,
                "internal_pressure_pa": p, "external_pressure_pa": 0.0}
        _, _, _, st, _ = lame_reference(torch.tensor([[R, 0.0, 0.0]], dtype=torch.float64), prob)
        devs.append(abs(float(st[0]) / (p * R / (2 * t)) - 1.0))
    # deviation is O(t/R) and must shrink with the ratio
    assert devs[0] < 2e-3 and devs[1] < devs[0]


# ------------------------------------------------------------------ production
def test_production_mesh_hits_membrane_state_within_3_percent():
    res = sphere_pressure(radius=CASE["radius"], thickness=CASE["thickness"],
                          young=CASE["young"], poisson=CASE["poisson"],
                          pressure=CASE["pressure"], facets=8, exact_geometry=True)
    sigma_ref = CASE["pressure"] * CASE["radius"] / (2 * CASE["thickness"])
    assert sigma_ref == pytest.approx(125.0)
    assert res["N_theta_max_rel_error"] < TOL
    assert res["N_phi_max_rel_error"] < TOL
    assert res["anisotropy_rel_error"] < TOL          # equi-biaxial, not just close on average
    assert res["w_rel_error"] < TOL
    # and it is not merely inside tolerance: it is at quadrature/round-off level
    assert res["N_theta_max_rel_error"] < 1e-5


def test_refinement_reduces_error_monotonically():
    errs = []
    for facets in (2, 4, 8, 16, 32):
        res = sphere_pressure(radius=CASE["radius"], thickness=CASE["thickness"],
                              young=CASE["young"], poisson=CASE["poisson"],
                              pressure=CASE["pressure"], facets=facets,
                              exact_geometry=True)
        errs.append(res["N_theta_max_rel_error"])
    assert all(a > b for a, b in zip(errs, errs[1:]))
    # observed order between the last two levels is at least 1.5
    order = math.log(errs[-2] / errs[-1]) / math.log(2.0)
    assert order >= 1.5


@pytest.mark.parametrize("nu", [0.0, 0.3, 0.49])
def test_poisson_invariance_of_membrane_stress(nu):
    """A closed sphere's membrane stress is nu-independent; w scales with (1-nu)."""
    res = sphere_pressure(radius=CASE["radius"], thickness=CASE["thickness"],
                          young=CASE["young"], poisson=nu,
                          pressure=CASE["pressure"], facets=8, exact_geometry=True)
    assert res["N_theta_max_rel_error"] < TOL
    assert res["w_rel_error"] < TOL


# ------------------------------------------------------- fail-closed controls
def test_chord_meridian_control_still_fails():
    """Straight generatrices (R_s = inf) cannot carry the sphere membrane state."""
    res = sphere_pressure(radius=CASE["radius"], thickness=CASE["thickness"],
                          young=CASE["young"], poisson=CASE["poisson"],
                          pressure=CASE["pressure"], facets=8, exact_geometry=False)
    assert res["N_theta_max_rel_error"] > TOL
    assert res["w_rel_error"] > TOL


@pytest.mark.slow
def test_flat_facet_shell4_control_still_fails():
    """The merged flat-facet Shell4 path misses pR/2 by two orders of magnitude."""
    import run_verification as rv

    flat = rv.flat_facet_shell4_control()
    assert flat["N_mean_rel_error"] > TOL
    assert flat["N_mean_rel_error"] > 10.0     # recorded 13317 % at 24 facets
    assert flat["nonradial_residual_rel"] > 0.5  # the response is not membrane at all


# ------------------------------------------------------------- stale evidence
def test_committed_evidence_matches_a_fresh_solve():
    """Guard against the committed JSON drifting away from the code."""
    path = os.path.join(CASE_DIR, "result.json")
    if not os.path.exists(path):
        pytest.skip("run_verification.py has not been run in this checkout")
    with open(path) as fh:
        stored = json.load(fh)
    result = stored["convergence"]["ladder"]["exact_meridian"]
    fresh = sphere_pressure(radius=CASE["radius"], thickness=CASE["thickness"],
                            young=CASE["young"], poisson=CASE["poisson"],
                            pressure=CASE["pressure"], facets=8, exact_geometry=True)
    row = next(r for r in result if r["facets"] == 8)
    assert row["sigma_theta_max_rel_error"] == pytest.approx(
        fresh["N_theta_max_rel_error"], rel=1e-9, abs=0.0)
    assert stored["reference"]["values"]["sigma_pa"] == pytest.approx(125.0, rel=1e-12)
    gates = stored["gates"]
    assert gates["all_gating_pass"] is True
    assert gates["falsified_predictions"] == ["G8b_reference_band_reaches_t_over_R_0.05"]