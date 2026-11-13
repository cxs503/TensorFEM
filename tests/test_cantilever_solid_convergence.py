"""Regression gate for the CASE-CANTILEVER-TIP solid-convergence anchor.

The anchor ``benchmarks/verified/fem_cantilever_solid_convergence`` grades the HEX20
cantilever against the library's *exact plane-stress elasticity* solution (not against a
beam theory), with the root boundary condition varied as a controlled experiment.

Two things are checked here:

* the committed evidence is internally consistent and every gate passes -- a gate whose
  value violates its own operator is a defect even if the JSON says ``passed: true``;
* the evidence is reproducible from the code: the cheapest recorded level is recomputed
  live and compared with the stored number, and the consistent tip traction is re-integrated
  to its resultant ``-P``.

Nothing here re-runs the expensive levels; the driver owns those.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
DRIVER_DIR = REPO / "benchmarks" / "verified" / "fem_cantilever_solid_convergence"

for p in (REPO / "src", REPO / "benchmarks" / "verified"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "fem_cantilever_solid_convergence_run", DRIVER_DIR / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver():
    return _load_driver()


@pytest.fixture(scope="module")
def result():
    return json.loads((DRIVER_DIR / "result.json").read_text())


@pytest.fixture(scope="module")
def manifest():
    return json.loads((DRIVER_DIR / "gates_verified.json").read_text())


def test_manifest_schema_and_verdict(manifest):
    assert manifest["schema"] == "tensorfem.verified-gates.v1"
    assert manifest["anchor"] == "fem_cantilever_solid_convergence"
    assert manifest["passed"] is True
    assert manifest["failed_gates"] == []
    assert manifest["gates"], "manifest carries no gates"


def test_every_gate_satisfies_its_own_criterion(manifest):
    """A gate is only green if re-evaluating its value against its operator passes."""
    for gate in manifest["gates"]:
        value, threshold, op = gate["value"], gate["threshold"], gate["operator"]
        ok = value <= threshold if op == "<=" else value >= threshold
        assert ok, f"gate {gate['id']} reports value={value} against {op} {threshold}"
        assert gate["passed"] is True, gate["id"]
        assert gate["reference"], f"gate {gate['id']} has no reference oracle"


def test_gate_matches_result(manifest, result):
    manifest_ids = [g["id"] for g in manifest["gates"]]
    result_ids = [g["id"] for g in result["gates"]]
    assert manifest_ids == result_ids
    assert result["passed"] is manifest["passed"]
    assert result["failed_gates"] == manifest["failed_gates"]


def test_exact_elasticity_oracle_matches_closed_form(driver):
    """The Airy module and the closed form used as the oracle must be the same object."""
    for slenderness in driver.SLENDERNESS:
        length = slenderness * driver.H
        closed = driver.delta_exact_closed_form(length)
        module = float(driver.AiryCantilever(
            length=length, height=driver.H, thickness=driver.B,
            young_modulus=driver.E, poisson=driver.NU, end_shear=-driver.P,
        ).tip_deflection())
        assert abs(module) == pytest.approx(closed, rel=1e-7)   # tip_deflection is signed


def test_kappa_equivalence_identity(driver, result):
    """Exact plane stress carries kappa = 2/3, not 5/6, at nu = 0.3."""
    ratio = driver.shear_ratio_exact_over_timoshenko()
    assert ratio == pytest.approx(1.25, rel=1e-12)
    assert driver.kappa_equivalent_exact_2d() == pytest.approx(2.0 / 3.0, rel=1e-12)
    assert result["oracle_selfcheck"]["kappa_equivalent_exact_2d"] == pytest.approx(
        driver.kappa_equivalent_exact_2d(), rel=1e-12)


def test_tip_traction_resultant_is_minus_p(driver):
    """Consistent parabolic end load must integrate to exactly -P and a zero moment."""
    length = 10.0 * driver.H
    nodes, elements = driver.structured_hex20_mesh(length, driver.B, driver.H, 6, 2, 2)
    forces = driver.tip_face_forces(nodes, elements, length, 2, 2)
    tip = torch.nonzero((nodes[:, 0] - length).abs() < 1e-12, as_tuple=False).reshape(-1)
    total = float(forces.reshape(-1, 3)[:, 2].sum())
    assert total == pytest.approx(-driver.P, rel=1e-10)
    # symmetric about the section mid-height => no net couple inside the end face
    couple = float(((nodes[:, 2] - driver.H / 2.0) * forces.reshape(-1, 3)[:, 2]).sum())
    assert abs(couple) < 1e-9 * driver.P * driver.H
    assert len(tip) > 0


def test_cheapest_recorded_level_reproduces(driver, result):
    """Recompute the cheapest recorded mesh level and bind it to the stored evidence."""
    level = min(result["mesh_study"]["levels"],
                key=lambda lv: lv["mesh"]["axial"] * lv["mesh"]["cross_y"] * lv["mesh"]["cross_z"])
    mesh = (level["mesh"]["axial"], level["mesh"]["cross_y"], level["mesh"]["cross_z"])
    live = driver.solid_tip(result["mesh_study"]["slenderness"] * driver.H, mesh, "axial_only")
    assert live["dofs"] == level["dofs"]
    assert live["tip_deflection_m"] == pytest.approx(level["tip_deflection_m"], rel=1e-9)
    exact = level["oracle_exact_2d_m"] if "oracle_exact_2d_m" in level else None
    if exact is not None:
        assert ((live["tip_deflection_m"] - exact) / exact
                == pytest.approx(level["error_vs_exact_2d"], rel=1e-9, abs=1e-12))


def test_root_condition_control_is_wellposed(driver, result):
    """axial_only carries exactly three more constraints than its complement and no less."""
    rows = result["ladder"]
    for row in rows:
        assert row["axial_only"]["n_constraints"] == row["clamp_all"]["n_constraints"] // 3 + 3
        assert row["axial_only"]["force_residual"] <= 1e-5
        assert row["axial_only"]["moment_residual"] <= 1e-5


def test_solid_meets_case_tolerance(result):
    """The case criterion (< 3 %) must hold against both the beam and elasticity oracles."""
    rows = result["ladder"]
    worst_ti = max(abs(r["axial_only"]["error_vs_timoshenko"]) for r in rows)
    worst_exact = max(abs(r["axial_only"]["error_vs_exact_2d"]) for r in rows
                      if r["slenderness"] >= 10.0)
    assert worst_ti <= result["problem"]["case_tolerance"]
    assert worst_exact <= 1e-2


def test_docs_report_injected_numbers(result):
    """The generated report must quote the computed numbers, not placeholders."""
    doc = (REPO / "docs" / "CASE-CANTILEVER-TIP-solid-convergence.md").read_text()
    worst = max(abs(r["axial_only"]["error_vs_exact_2d"]) for r in result["ladder"])
    assert f"{1e2 * worst:.3f}" in doc
    assert "PASS" in doc and "none" not in doc.split("Overall:")[1][:40].lower()
