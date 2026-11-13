"""CASE-CANTILEVER-TIP -- guards for the pure-moment bending-flexibility bundle.

The numbers live in ``benchmarks/verified/fem_cantilever_shear/bending_recovery.json``.
These tests re-derive every published fit from the stored sweep (the fits cost nothing)
and re-run one cheap solve to keep the measurement path itself under test.  The verdict
of the bundle is an honest FAIL (the pinned two-parameter refit does not identify the
shear coefficient); the tests assert that verdict is *recorded as such* rather than
silently upgraded.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmarks" / "verified" / "fem_cantilever_shear"
EVIDENCE = BENCH / "bending_recovery.json"
FORCE = BENCH / "kappa_recovery.json"
DOC = REPO / "docs" / "CASE-CANTILEVER-TIP-pinned-bending.md"
FIG = REPO / "docs" / "assets" / "cantilever_shear" / "bending_recovery.png"


@pytest.fixture(scope="module")
def evidence() -> dict:
    if not EVIDENCE.exists():
        pytest.skip("bending_recovery.json not generated")
    return json.loads(EVIDENCE.read_text())


@pytest.fixture(scope="module")
def module():
    for extra in (BENCH, REPO / "src", REPO / "benchmarks" / "verified"):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    import bending_recovery

    return bending_recovery


def test_artifacts_present(evidence):
    assert DOC.exists() and FIG.exists()
    assert evidence["anchor"] == "fem_cantilever_shear"
    assert evidence["mission"] == "CASE-CANTILEVER-TIP"
    assert len(evidence["sweep"]) == len(evidence["problem"]["fit_slenderness"])


def test_symbolic_oracle_is_clean(evidence, module):
    """The pure-bending field and the slope identity must be derived, not quoted."""
    oracle = module.pure_bending_oracle()
    assert oracle["all_symbolic_checks_zero"] is True
    assert oracle["biharmonic_residual"] == "0"
    assert oracle["curvature_identity"] == "0"
    # the field carries no net axial force and exactly the applied couple
    assert oracle["net_axial_force"] == "0"
    assert oracle["net_end_moment"] == "M"
    assert evidence["pure_bending_oracle"]["all_symbolic_checks_zero"] is True


def test_load_is_a_pure_couple_on_a_coarse_mesh(module):
    """No solve needed: the consistent tip loads must carry zero force and exactly M."""
    nodes, elements = module.structured_hex20_mesh(0.04, module.B, module.H, 6, 2, 2)
    forces, check = module.tip_face_loads(nodes, elements, 0.04)
    assert check["tip_face_elements"] == 4
    assert check["net_axial_force_relative"] < 1e-12
    assert check["applied_moment_relative_error"] < 1e-12
    # only axial components are loaded
    assert float(forces.reshape(-1, 3)[:, 1:].abs().sum()) == 0.0


def test_moment_ladder_matches_the_exact_slope(evidence):
    """Every rung of the pure-moment ladder, against theta = M L/(E I)."""
    worst = 0.0
    for row in evidence["sweep"]:
        exact = row["moment_Nm"] * row["length_m"] / (E_REF * I_REF)
        assert row["slope_exact"] == pytest.approx(exact, rel=1e-12)
        assert row["slope_measured"] / exact == pytest.approx(row["slope_ratio"], rel=1e-12)
        worst = max(worst, abs(row["slope_ratio"] - 1.0))
    assert worst < 0.03, f"pure-moment slope error {worst*100:.3f}% exceeds the case tolerance"


E_REF = 200.0e9
I_REF = 0.01 * 0.01**3 / 12.0


def test_slope_fit_is_rederivable_from_the_stored_sweep(evidence, module):
    fit = module.fit_slope_ratio(evidence["sweep"])
    recorded = evidence["slope_fit"]
    for key in ("a_moment", "d_root_layer", "r2", "sigma_a", "max_abs_residual"):
        assert fit[key] == pytest.approx(recorded[key], rel=1e-9, abs=1e-15), key
    assert recorded["sigma_a"] > 0.0
    assert recorded["design_condition_number"] < 100.0  # 2-parameter design, 5x span in h/L


def test_pinned_refit_is_rederivable_and_its_failure_is_recorded(evidence, module):
    fit = module.refit_force_sweep(evidence["slope_fit"]["a_moment"], evidence["slope_fit"]["sigma_a"])
    recorded = evidence["force_refit"]
    for key in ("b_shear_ratio", "kappa_eff", "sigma_kappa", "max_abs_residual", "r2"):
        assert fit[key] == pytest.approx(recorded[key], rel=1e-9, abs=1e-15), key
    # the pre-registered verdict is an honest FAIL: the refit does not identify kappa
    assert recorded["max_abs_residual"] > 1e-3
    assert recorded["kappa_eff"] > 1.0  # unphysical kappa > 1 is the symptom, not a result
    assert evidence["all_gates_pass"] is False
    failed = {g["name"] for g in evidence["gates"] if not g["pass"]}
    assert {"pinned_fit_residual_below_0p1pct",
            "sigma_kappa_shrinks_vs_three_parameter_fit"} <= failed


def test_force_sweep_it_consumes_is_reproducible(evidence):
    """The stored three-parameter fit must still sit on the stored force sweep."""
    force = json.loads(FORCE.read_text())["fit"]
    rows = json.loads(FORCE.read_text())["sweep"]["fit"]
    x = np.array([0.01 / r["length_m"] for r in rows])
    e = np.array([r["error_vs_eb"] for r in rows])
    pred = force["A"] + force["B"] * x**2 + force["C"] * x**3
    assert np.max(np.abs(e - pred)) <= 1e-3
    for row in evidence["force_crosscheck"]:
        assert abs(row["difference"]) < 1e-9


def test_live_pure_moment_solve_on_a_coarse_mesh(module):
    """One cheap solve (no cache): the measurement path must reproduce theta = M L/(E I)."""
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    row = module.pure_moment_solid(0.04, (24, 1, 1))
    assert row["reaction_moment_residual"] < 1e-8
    assert row["applied_moment_relative_error"] < 1e-12
    assert abs(row["slope_ratio"] - 1.0) < 0.03, f"slope error {(row['slope_ratio']-1)*100:.3f}%"


def test_mesh_crosscheck_rows_are_present(evidence):
    cross = evidence["mesh_crosscheck"]
    assert [r["mesh"]["cross_y"] for r in cross] == [2, 3]
    # the section mesh moves the measured flexibility by less than 0.1 pp
    spread = abs(cross[0]["slope_ratio"] - cross[1]["slope_ratio"])
    assert spread < 1e-3