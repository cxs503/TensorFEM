"""CASE-SPHERE-PRESSURE — module-level gates for the production case code.

Locks :mod:`tensorfem.sphere_pressure_case` (the single source of truth for this
case pool entry) into the test suite:

* the reference is *derived* (sympy axial equilibrium + Laplace equilibrium with
  exactly zero residual), never transcribed, and its value is ``pR/(2t)``;
* on the production axisymmetric shell of revolution with circular-arc
  meridians the solved state hits the membrane state well inside the
  predeclared 3 % tolerance (hoop, meridional, isotropy, radial displacement);
* the case is wired into the tree: registry callers, console-script entry,
  thin runner script — so it cannot regress into a free-floating document.

Runtime: a few seconds for the fast tests; the CLI end-to-end test is marked
``slow`` because it runs the full study bundle (~15 s).
"""
from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensorfem import sphere_pressure_case as case_mod  # noqa: E402


# --------------------------------------------------------------- reference --
def test_reference_is_derived_and_equals_pR_over_2t():
    ref = case_mod.reference()
    par = case_mod.CASE
    sigma_expected = par["pressure_pa"] * par["radius_m"] / (2.0 * par["thickness_m"])
    w_expected = (par["pressure_pa"] * par["radius_m"] ** 2 *
                  (1.0 - par["poisson_ratio"]) / (2.0 * par["young_modulus_pa"] *
                                                  par["thickness_m"]))
    assert ref["residual_is_zero"] is True
    # sympy may print either R*p/(2*t) or p*r/(2*t); both are the same formula
    assert ref["derivation"]["sigma_expr"].replace(" ", "") in ("R*p/(2*t)", "p*r/(2*t)")
    assert ref["values"]["sigma_pa"] == pytest.approx(sigma_expected, rel=1e-14)
    assert ref["values"]["w_m"] == pytest.approx(w_expected, rel=1e-14)
    # the formula recomputed without sympy must agree with the symbolic result
    assert ref["value_checks"]["sigma_from_formula"] == pytest.approx(
        ref["values"]["sigma_pa"], rel=1e-14)


# ------------------------------------------------------------------ solver --
def test_production_mesh_meets_3_percent_gate():
    q = case_mod.quick_check()
    assert q["pass"] is True
    assert q["sigma_theta_rel_error"] < case_mod.TOL
    assert q["sigma_phi_rel_error"] < case_mod.TOL
    assert q["anisotropy_rel"] < case_mod.TOL
    assert q["w_rel_error"] < case_mod.TOL
    # the membrane state must be equi-biaxial, not merely close on one component
    assert q["sigma_theta_pa"] == pytest.approx(q["sigma_phi_pa"], rel=1e-4)


def test_chord_meridian_control_stays_a_failure():
    """Geometry-representation control: straight chords must miss by >> 3 %."""
    chord = case_mod.chord_control(case_mod.PRODUCTION_FACETS)
    assert chord["sigma_theta_max_rel_error"] > case_mod.TOL
    exact = case_mod.axisym_case(case_mod.PRODUCTION_FACETS, exact_geometry=True)
    assert exact["sigma_theta_max_rel_error"] < chord["sigma_theta_max_rel_error"]


# -------------------------------------------------------------- wiring -----
def test_registry_has_real_callers():
    from tensorfem import benchmark_registry as reg

    src = open(os.path.join(ROOT, "src", "tensorfem", "benchmark_registry.py")).read()
    assert "from .sphere_pressure_case import quick_check" in src
    for case_id in ("shell.sphere.pressure_sigma_theta",
                    "shell.sphere.pressure_sigma_phi",
                    "shell.sphere.pressure_radial_displacement"):
        assert case_id in src
    # the function the registry imports must exist and be callable
    assert callable(reg.run_registered_benchmarks)


def test_console_script_and_runner_are_declared():
    pyproject = open(os.path.join(ROOT, "pyproject.toml")).read()
    assert 'tensorfem-sphere-pressure = "tensorfem.sphere_pressure_case:cli"' in pyproject
    assert callable(case_mod.cli)
    runner = os.path.join(ROOT, "benchmarks", "verified",
                          "fem_sphere_pressure_membrane", "run_verification.py")
    text = open(runner).read()
    # the runner must stay a thin wrapper: it delegates, it does not re-implement
    assert "from tensorfem.sphere_pressure_case import" in text
    assert "def reference(" not in text


@pytest.mark.slow
def test_cli_end_to_end_writes_artifacts(tmp_path):
    out = tmp_path / "CASE-SPHERE-PRESSURE"
    rc = case_mod.cli(["--out", str(out), "--no-figures", "--quiet"])
    assert rc == 0
    result = json.loads((out / "result.json").read_text())
    gates = json.loads((out / "gates_verified.json").read_text())
    assert result["case"] == "CASE-SPHERE-PRESSURE"
    assert gates["all_gating_pass"] is True
    assert result["tolerance"] == 0.03
    ref = result["reference"]["values"]["sigma_pa"]
    row = next(r for r in result["convergence"]["ladder"]["exact_meridian"]
               if r["facets"] == case_mod.PRODUCTION_FACETS)
    got = max(row["sigma_theta_pa"])
    assert abs(got - ref) / ref < 0.03
    assert result["repro_cmd"].startswith("tensorfem-sphere-pressure")