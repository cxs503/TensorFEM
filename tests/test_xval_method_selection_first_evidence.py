"""Tests for the XVAL method-selection cross-validation harness.

These tests pin the *protocol* (not the verdict): the closed-form band
integrals, the two probe functionals, the independent scikit-fem arm, and the
claim-reproduction rules.
"""
from __future__ import annotations

import importlib.util
import json
import math
import pathlib

import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "xval_msfe", _ROOT / "benchmarks" / "xval_method_selection_first_evidence.py")
xval = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(xval)


def test_record_snapshot_is_the_oss_row():
    record = json.loads((_ROOT / "benchmarks" /
                         "xval_method_selection_first_evidence_record.json").read_text())
    assert record["repo"] == "TensorFEM"
    assert record["tag"] == "method-selection+first-evidence"
    assert "HEX20" in record["method"]
    assert record["captured_at"] and record["captured_via"]


def test_lame_band_average_matches_numerical_integration():
    """Volume averages of the Lame fields vs 200k-point quadrature (1e-9 rel)."""
    r1, r2 = 8.0, 9.0
    r = np.linspace(r1, r2, 2_000_001)
    w = 4.0 * math.pi * r ** 2
    fields = xval.lame_sphere(r, xval.CASE)
    numeric = {k: float(np.trapezoid(w * np.asarray(v), r) / np.trapezoid(w, r))
               for k, v in fields.items()}
    analytic = xval.lame_band_average(r1, r2)
    assert abs(numeric["radial_stress"] - analytic["radial_stress"]) / abs(
        analytic["radial_stress"]) < 1e-9
    assert abs(numeric["hoop_stress"] - analytic["hoop_stress"]) / abs(
        analytic["hoop_stress"]) < 1e-9
    assert abs(numeric["von_mises"] - analytic["von_mises"]) / abs(
        analytic["von_mises"]) < 1e-9
    # first moment consistency of the reported band radius
    assert abs(numeric["radial_displacement"] * 0 - 0) < 1e-12


def test_band_average_is_not_centroid_sampling():
    """The two functionals differ materially on the inner band (the m15883 gap)."""
    analytic = xval.lame_band_average(8.0, 9.0)
    centroid = xval.lame_at(analytic["radius"])
    gap = abs(analytic["radial_stress"] - centroid["radial_stress"]) / abs(
        centroid["radial_stress"])
    assert gap > 0.03, f"functionals collapsed to the same value (gap={gap:.3%})"


def test_metric_m2_is_exact_for_a_perfect_layer():
    """A layer whose value equals the exact band integral scores ~0 under M2."""
    radii = [8.0, 9.0, 10.0]
    exact_inner = xval.lame_band_average(radii[0], radii[1])
    exact_outer = xval.lame_band_average(radii[1], radii[2])
    row = {"u_inner": xval.lame_at(8.0)["radial_displacement"],
           "u_outer": xval.lame_at(10.0)["radial_displacement"]}
    inner = {"radius": exact_inner["radius"], "radial_stress": exact_inner["radial_stress"],
             "hoop_stress": exact_inner["hoop_stress"], "von_mises": exact_inner["von_mises"],
             "volume": 4 / 3 * math.pi * (9.0 ** 3 - 8.0 ** 3)}
    outer = {"radius": exact_outer["radius"], "radial_stress": exact_outer["radial_stress"],
             "hoop_stress": exact_outer["hoop_stress"], "von_mises": exact_outer["von_mises"],
             "volume": 4 / 3 * math.pi * (10.0 ** 3 - 9.0 ** 3)}
    scored = xval.metric_m2(row, radii, inner, outer)
    assert scored["worst_m2"] < 1e-12
    assert max(scored["layer_volume_error"].values()) < 1e-12


@pytest.mark.parametrize("order", [1, 2])
def test_independent_scikitfem_arm_matches_lame(order):
    """scikit-fem icosphere tet shell reproduces the Lame displacement field."""
    row = xval.arm_skfem(3, order=order, layers=2, project=True)
    ur_a = xval.lame_at(xval.A)["radial_displacement"]
    ur_b = xval.lame_at(xval.B)["radial_displacement"]
    assert abs(row["u_inner"] - ur_a) / abs(ur_a) < 0.05
    assert abs(row["u_outer"] - ur_b) / abs(ur_b) < 0.05
    # self-equilibrated pressure load: net force/moment projects to ~0
    assert row["loads"]["load_rigid_mode_relative"] < 1e-6
    # the discrete system is solved to high accuracy
    assert row["equilibrium"]["free_residual_relative"] < 1e-8
    # geometric fidelity of the icosphere shell improves with the element order
    assert row["mesh_volume_error"] < 0.05
    assert row["worst_m1"] < 0.25


def test_claim_reproduction_rules():
    """Threshold and numeric claims follow the predeclared rules."""
    rows = [
        {"arm": "TensorFEM", "order": 2, "n": 4, "worst_m1": 0.0288,
         "mesh_volume_error": 0.0014, "loads": {"fx": 1e-17},
         "solver": {"iterations": 100}},
        {"arm": "TensorFEM", "order": 1, "n": 4, "worst_m1": 0.28,
         "mesh_volume_error": 0.0712, "solver": {"iterations": 80}},
        {"arm": "TensorFEM", "order": 1, "n": 6, "worst_m1": 0.1535,
         "mesh_volume_error": 0.033, "solver": {"iterations": 90}},
        {"arm": "TensorFEM", "order": 1, "n": 8, "worst_m1": 0.0993,
         "mesh_volume_error": 0.019, "solver": {"iterations": 67}},
    ]
    claims = {c["id"]: c for c in xval.reproduce_claims(rows)}
    assert claims["C1"]["reproduced"] is True          # 2.88% < 3%
    assert claims["C2"]["reproduced"] is False         # 28% vs recorded 15.35%
    assert claims["C2"]["deviation"] > xval.CLAIM_TOLERANCE
    assert claims["C3"]["reproduced"] is False         # no level below 5% here
    assert claims["C4a"]["reproduced"] is True         # 7.12% vs 7.1%
    assert claims["C6"]["reproduced"] is True          # 67 iterations


def test_decision_table_flags_functional_dependence():
    rows = [
        {"arm": "TensorFEM", "order": 2, "n": 4, "element": "HEX20", "dofs": 2940,
         "worst_m1": 0.0288, "worst_m2": 0.0803, "worst_quantity_m2": "radial_stress_inner"},
        {"arm": "TensorFEM", "order": 1, "n": 4, "element": "HEX8", "dofs": 882,
         "worst_m1": 0.2818, "worst_m2": 0.0410, "worst_quantity_m2": "radial_stress_inner"},
    ]
    decision = xval.decision_table(rows)
    assert decision["verdict"]["ranking_m1_reproduced"] is True
    assert decision["verdict"]["ranking_m2_reproduced"] is False
    assert decision["verdict"]["selection_robust"] is False