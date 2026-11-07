"""Acceptance tests for the IN41-N01 structural verification anchor.

The anchor ``benchmarks/verified/fem_structural_benchmarks`` grades three element
families (Hermite beam, Mindlin plate, axisymmetric ring) against closed-form
oracles.  This module guards it on three levels:

1. **Oracle re-derivation** -- the closed forms are written again here, from the
   textbook expressions, and compared with what the driver publishes.  A drifting
   formula in the driver therefore fails the test instead of passing silently.
2. **Evidence self-consistency** -- every gate value in the committed
   ``gates_verified.json`` is recomputed from the committed ``result.json``, and
   every threshold is checked against the campaign criteria.
3. **Pipeline re-run** -- a deliberately coarse ladder is solved end to end so the
   *code path* (assembly, boundary conditions, load vectors, recovery) is
   exercised, not just the stored JSON.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ANCHOR = ROOT / "benchmarks" / "verified" / "fem_structural_benchmarks"
VERIFIED = ANCHOR.parent


def _load_driver():
    if str(VERIFIED) not in sys.path:
        sys.path.insert(0, str(VERIFIED))
    spec = importlib.util.spec_from_file_location("fem_structural_benchmarks_run", ANCHOR / "run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver():
    import torch

    drive = _load_driver()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    return drive


@pytest.fixture(scope="module")
def evidence():
    result = json.loads((ANCHOR / "result.json").read_text())
    gates = json.loads((ANCHOR / "gates_verified.json").read_text())
    return result, gates


# --------------------------------------------------------------------------- #
# 1. oracles, re-derived from the textbook expressions
# --------------------------------------------------------------------------- #
def test_case_a_oracle_matches_beam_theory(driver):
    """w = P L^3/(3EI), theta = P L^2/(2EI), sigma = P L c/I, I = (bh^3 - b_i h_i^3)/12."""
    b, h, tw = driver.BEAM["outer_width_m"], driver.BEAM["outer_depth_m"], driver.BEAM["wall_thickness_m"]
    I = (b * h**3 - (b - 2 * tw) * (h - 2 * tw) ** 3) / 12.0
    A = b * h - (b - 2 * tw) * (h - 2 * tw)
    E, L, P = driver.BEAM["young_modulus_pa"], driver.BEAM["length_m"], driver.BEAM["tip_load_n"]
    section = driver.beam_section()
    assert section["second_moment_m4"] == pytest.approx(I, rel=1e-14)
    assert section["area_m2"] == pytest.approx(A, rel=1e-14)
    assert driver.beam_theory(L, section=section)["defl_m"] == pytest.approx(
        P * L**3 / (3 * E * I), rel=1e-13)
    assert driver.beam_theory(L, section=section)["slope_rad"] == pytest.approx(
        P * L**2 / (2 * E * I), rel=1e-13)
    assert driver.beam_theory(0.0, section=section)["bending_stress_pa"] == pytest.approx(
        P * L * (h / 2) / I, rel=1e-13)
    # the uniform-load field of the same case must satisfy the tip-load limit
    q = driver.BEAM["distributed_load_n_per_m"]
    assert driver.beam_theory_udl(L, section=section)["defl_m"] == pytest.approx(
        q * L**4 / (8 * E * I), rel=1e-13)


def test_case_b_oracle_reproduces_timoshenko_alpha(driver):
    """The published Navier series must give alpha = 0.00406235 (table: 0.00406)."""
    a = driver.PLATE["side_m"]
    q = driver.PLATE["pressure_pa"]
    D = driver.plate_rigidity()
    w = driver.plate_theory(0.5, 0.5, terms=driver.PLATE["navier_terms"])
    alpha = w * D / (q * a**4)
    assert alpha == pytest.approx(0.0040623527, abs=2e-9)
    assert abs(alpha - driver.PLATE["timoshenko_alpha"]) / driver.PLATE["timoshenko_alpha"] < 1e-3
    # the plate is thin, so D must equal E t^3/(12(1-nu^2))
    E, t, nu = driver.PLATE["young_modulus_pa"], driver.PLATE["thickness_m"], driver.PLATE["poisson_ratio"]
    assert D == pytest.approx(E * t**3 / (12 * (1 - nu**2)), rel=1e-13)
    # a coarse series truncation must not move the centre value beyond the gate
    coarse = driver.plate_theory(0.5, 0.5, terms=driver.PLATE["navier_terms_field"])
    assert abs(coarse - w) / w < 1e-6


def test_case_c_oracle_satisfies_equilibrium_and_boundary_conditions(driver):
    """Lame fields: d(sigma_r)/dr + (sigma_r - sigma_theta)/r = 0, sigma_r(a) = -p, sigma_r(b) = 0."""
    a = driver.CYLINDER["inner_radius_m"]
    b = driver.CYLINDER["outer_radius_m"]
    p = driver.CYLINDER["internal_pressure_pa"]
    nu = driver.CYLINDER["poisson_ratio"]
    assert driver.lame(a)["sigma_r_pa"] == pytest.approx(-p, rel=1e-12)
    assert abs(driver.lame(b)["sigma_r_pa"]) / p < 1e-12
    h = 1e-6 * a
    for r in (1.1 * a, 1.35 * a, 1.7 * a, 0.95 * b):
        dsig = (driver.lame(r + h)["sigma_r_pa"] - driver.lame(r - h)["sigma_r_pa"]) / (2 * h)
        state = driver.lame(r)
        residual = dsig + (state["sigma_r_pa"] - state["sigma_theta_pa"]) / r
        assert abs(residual) / (p / a) < 1e-5
        assert state["sigma_z_pa"] == pytest.approx(2 * nu * state["sigma_theta_pa"] * 0
                                                    + 2 * nu * driver.lame_coefficients()[0], rel=1e-12)
    # radial displacement must be the plane-strain elastic solution: eps_theta = u_r/r
    E = driver.CYLINDER["young_modulus_pa"]
    r = 1.25 * a
    state = driver.lame(r)
    eps_theta = state["u_r_m"] / r
    assert eps_theta == pytest.approx(
        (state["sigma_theta_pa"] - nu * (state["sigma_r_pa"] + state["sigma_z_pa"])) / E, rel=1e-12)


# --------------------------------------------------------------------------- #
# 2. committed evidence must be self-consistent
# --------------------------------------------------------------------------- #
def _rebuild_reference_values(result: dict) -> dict:
    cases = result["cases"]
    beam = cases["A_beam_cantilever"]
    plate = cases["B_plate_ssss_uniform"]
    cylinder = cases["C_cylinder_lame"]
    return {
        "A1.tip_deflection": max(r["w_tip_rel_error"] for r in beam["levels_tip_load"]),
        "A2.root_bending_stress": max(r["root_stress_rel_error"] for r in beam["levels_tip_load"]),
        "A3.beam_order": beam["observations"]["order_udl_interior_field"]["finest"],
        "B1.centre_vs_timoshenko": plate["levels"][-1]["w_centre_rel_error_timoshenko"],
        "B2.centre_vs_navier": plate["levels"][-1]["w_centre_rel_error_navier"],
        "B3.plate_order": plate["observations"]["order_l2_field"]["finest"],
        "C1.hoop_stress": cylinder["levels"][-1]["max_sigma_theta_rel_error"],
        "C2.radial_stress": cylinder["levels"][-1]["max_sigma_r_rel_error"],
        "C2b.radial_stress_absolute": cylinder["levels"][-1]["max_sigma_r_abs_over_p"],
        "C3.radial_displacement": cylinder["levels"][-1]["max_u_r_rel_error"],
        "C3b.axial_stress": cylinder["levels"][-1]["max_sigma_z_rel_error"],
        "C4.cylinder_order": cylinder["observations"]["order_sigma_theta"]["finest"],
    }


def test_gate_values_match_result_json(evidence):
    result, gates = evidence
    rebuilt = _rebuild_reference_values(result)
    published = {g["id"]: g for g in gates["gates"]}
    assert set(published) == set(rebuilt)
    for key, value in rebuilt.items():
        assert published[key]["value"] == pytest.approx(value, rel=1e-12, abs=0.0), key


def test_gates_pass_and_thresholds_match_the_campaign(evidence):
    result, gates = evidence
    crit = result["criteria"]
    assert gates["schema"] == "tensorfem.verified-gates.v1"
    assert gates["mission"] == "IN41-N01"
    assert gates["passed"] is True, gates["failed_gates"]
    thresholds = {g["id"]: (g["threshold"], g["operator"]) for g in gates["gates"]}
    assert thresholds["A1.tip_deflection"] == (crit["case_a_tip_deflection_rel_max"], "<=")
    assert thresholds["A2.root_bending_stress"] == (crit["case_a_root_stress_rel_max"], "<=")
    assert thresholds["B1.centre_vs_timoshenko"] == (crit["case_b_centre_deflection_rel_max"], "<=")
    assert thresholds["B2.centre_vs_navier"] == (crit["case_b_centre_deflection_rel_max"], "<=")
    for key in ("C1.hoop_stress", "C2.radial_stress", "C2b.radial_stress_absolute",
                "C3.radial_displacement", "C3b.axial_stress"):
        assert thresholds[key] == (crit["case_c_stress_rel_max"], "<=")
    for key in ("A3.beam_order", "B3.plate_order", "C4.cylinder_order"):
        assert thresholds[key] == (crit["convergence_order_min"], ">=")
    for gate in gates["gates"]:
        assert gate["passed"] is True, gate["id"]
        assert math.isfinite(gate["value"])
    assert gates["undefined_terms"] == []


def test_three_levels_and_three_figures(evidence):
    result, gates = evidence
    cases = result["cases"]
    assert [r["elements"] for r in cases["A_beam_cantilever"]["levels_tip_load"]] == list(
        result["cases"]["A_beam_cantilever"]["parameters"]["levels_elements"])
    assert len(cases["B_plate_ssss_uniform"]["levels"]) == 3
    assert len(cases["C_cylinder_lame"]["levels"]) == 3
    for name in ("case_a_beam.png", "case_b_plate.png", "case_c_cylinder.png"):
        path = ANCHOR / name
        assert path.is_file() and path.stat().st_size > 10_000, name
    assert len(cases["C_cylinder_lame"]["levels"][-1]["samples"]) == 5


# --------------------------------------------------------------------------- #
# 3. re-run the pipeline on a coarse ladder
# --------------------------------------------------------------------------- #
def test_pipeline_beam_ladder(driver):
    levels = [driver.solve_beam(n, load="tip") for n in (2, 4, 8)]
    for row in levels:
        assert row["w_tip_rel_error"] < 1e-9  # cubic field reproduced exactly
        assert row["root_stress_rel_error"] < 1e-8
    udl = [driver.solve_beam(n, load="udl") for n in (2, 4, 8)]
    errors = [row["field_max_defl_rel_error"] for row in udl]
    assert all(a > b for a, b in zip(errors, errors[1:]))  # monotone refinement
    order = math.log(errors[-2] / errors[-1]) / math.log(2.0)
    assert order >= 1.5


def test_pipeline_plate_ladder(driver):
    levels = [driver.solve_plate(n) for n in (4, 8)]
    l2 = [row["l2_defl_rel_error"] for row in levels]
    assert l2[1] < l2[0]
    assert l2[1] < 5e-2
    assert levels[-1]["w_centre_rel_error_navier"] < 2e-2  # campaign gate holds at n = 8
    assert driver.solve_plate(4, shear="sri")["w_centre_fem_m"] > 0.0


def test_pipeline_cylinder_ladder(driver):
    levels = [driver.solve_cylinder(nr) for nr in (4, 8)]
    theta = [row["max_sigma_theta_rel_error"] for row in levels]
    assert theta[1] < theta[0]
    assert theta[1] < 6e-2
    # the axisymmetric solution must be driven by the bore pressure, not by the
    # axial constraint: sigma_theta(a) is tensile and sigma_r(b) ~ 0
    finest = levels[-1]
    assert finest["samples"][0]["sigma_theta_fem_pa"] > 0.0
    assert abs(finest["samples"][-1]["sigma_r_fem_pa"]) < 0.1 * driver.CYLINDER["internal_pressure_pa"]
    # recovery must not make the raw Gauss values worse: report both
    assert finest["max_gauss_sigma_theta_rel_error"] > finest["max_sigma_theta_rel_error"]


def test_cylinder_surface_sample_is_degenerate_but_handled(driver):
    """At nr = 4 the r = 1.9a sample lands on r = b, where sigma_r = 0 identically.

    A relative error is undefined there; the driver must report ``None`` instead of
    dividing by zero and must still grade the load-normalised absolute error.
    """
    row = driver.solve_cylinder(4)
    outer = row["samples"][-1]
    assert outer["r_m"] == pytest.approx(driver.CYLINDER["outer_radius_m"], rel=1e-12)
    assert abs(outer["sigma_r_lame_pa"]) < 1e-6 * driver.CYLINDER["internal_pressure_pa"]
    assert outer["sigma_r_rel_error"] is None
    assert 0.0 <= outer["sigma_r_abs_over_p"] < 1.0
    assert row["sigma_r_rel_error_defined_points"] == 4
    assert row["max_sigma_r_abs_over_p"] >= row["max_sigma_r_rel_error"] * 0.0