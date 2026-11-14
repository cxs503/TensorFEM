"""Tests for the KIRSCH-PLATE three-dimensional (HEX8) verification anchor.

Three layers of assertion:

* a **live, cheap solve** (12x6x2 elements, 819 dofs) that must already land
  inside the 5 % case-pool tolerance and must produce ``sigma_zz ~ 0`` — this
  protects the physics and the 3D assembly path, not just the JSON;
* the **oracle self-check**: the claimed exact 3D solution really is divergence
  free and traction free on the hole edge;
* the **committed evidence**: ``solid3d_result.json`` / ``solid3d_gates.json``
  agree with each other, every gate passes, and the sha256 sidecar matches, so
  an edited number fails the suite instead of passing silently.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "verified" / "fem_kirsch_plate"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from tensorfem.kirsch_hole import KirschHole  # noqa: E402
from tensorfem.solid3d import hex8_stiffness, elasticity_matrix_3d, _hex_gradient  # noqa: E402

import solid3d_verification as S  # noqa: E402

HOLE = KirschHole(radius=1.0, far_field=1.0, young=1.0, poisson=0.3, plane="stress")


# --------------------------------------------------------------------------
# geometry / mesh
# --------------------------------------------------------------------------
def test_mesh_is_positive_and_well_formed():
    nodes, hexes, meta = S.extrude_quarter_annulus(HOLE, 4.0, 12, 6, 0.05, 2)
    assert nodes.shape == (13 * 7 * 3, 3)      # (n_theta+1)(n_radial+1)(n_layers+1)
    assert hexes.shape == (12 * 6 * 2, 8)
    assert int(hexes.max()) < nodes.shape[0]
    # every hex must have a positive Jacobian (hex8_stiffness fails closed otherwise)
    d = elasticity_matrix_3d(nodes.new_full((len(hexes),), 1.0), nodes.new_full((len(hexes),), 0.3))
    ke, _ = hex8_stiffness(nodes[hexes], d)
    assert torch.isfinite(ke).all()
    _, det = _hex_gradient(nodes[hexes], 0.0, 0.0, 0.0)
    assert float(det.min()) > 0.0
    # in-plane extent: hole radius on the inner arc, outer radius on the outer arc
    r = torch.linalg.vector_norm(nodes[:, :2], dim=1).numpy()
    assert r.min() == pytest.approx(1.0, rel=1e-12)
    assert r.max() == pytest.approx(4.0, rel=1e-12)


# --------------------------------------------------------------------------
# oracle
# --------------------------------------------------------------------------
def test_exact_3d_solution_is_equilibrium_and_traction_free():
    check = S.oracle_selfcheck(HOLE)
    assert check["hole_edge_traction_over_T"] < 1e-9
    assert check["divergence_max_over_T"] < 1e-6
    assert check["out_of_plane_stress_over_T"] == 0.0
    assert check["far_field_sxx_over_T"] == pytest.approx(1.0, rel=1e-3)
    assert abs(check["far_field_syy_over_T"]) < 1e-2


def test_exact_uz_is_plane_stress_kinematics():
    # sigma_xx + sigma_yy is harmonic-free in the far field: check the closed form
    # u_z = -nu/E (sigma_xx + sigma_yy) z is linear in z with the right slope
    x, y = 3.0, 0.5
    z = 0.05
    uz = S.exact_uz(HOLE, x, y, z)
    sxx, syy, _ = HOLE.stress_cartesian(x, y)
    assert uz == pytest.approx(-HOLE.poisson / HOLE.young * (sxx + syy) * z)
    assert S.exact_uz(HOLE, x, y, 0.0) == 0.0


# --------------------------------------------------------------------------
# live coarse solve (fast: 819 dofs)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def coarse():
    return S.solve_level(HOLE, 12, 6, 2)


def test_coarse_kt_already_inside_case_pool_tolerance(coarse):
    # 12x6x2 elements only: the anchor must already be inside 5 % on the coarsest
    # mesh, otherwise the committed ladder's first rung is not evidence
    assert coarse["peak"]["rel_error"] < 0.05
    assert 2.85 <= coarse["peak"]["kt_estimate"] <= 3.15


def test_coarse_szz_is_small(coarse):
    # exact solution has sigma_zz = 0; a locking-prone HEX8 would show much more
    assert coarse["field"]["szz_max_over_T"] < 0.05
    assert coarse["field"]["szz_rms_over_T"] < 0.05


def test_coarse_uz_matches_plane_stress_kinematics(coarse):
    assert coarse["uz"]["uz_rel_l2"] < 0.10


def test_coarse_pointwise_probes_are_finite_and_reasonable(coarse):
    for probe in coarse["pointwise"]:
        assert len(probe["exact"]) == len(probe["fe"]) == 6
        assert np.isfinite(probe["fe"]).all()
    assert coarse["pointwise_max_rel"] < 0.25   # coarse mesh, still bounded


def test_cg_tolerance_used_is_declared():
    """The solve uses the declared RTOL; rtol is part of the evidence contract."""
    assert S.RTOL == pytest.approx(1e-9)
    assert "rtol" in S.solve_case.__code__.co_varnames


# --------------------------------------------------------------------------
# committed evidence
# --------------------------------------------------------------------------
def test_committed_artifacts_exist_and_are_consistent():
    result_path = BENCH / "solid3d_result.json"
    gates_path = BENCH / "solid3d_gates.json"
    sha_path = BENCH / "solid3d_result.json.sha256"
    for path in (result_path, gates_path, sha_path):
        assert path.exists(), f"missing evidence {path}"
    payload = result_path.read_text()
    assert hashlib.sha256(payload.rstrip("\n").encode()).hexdigest() == sha_path.read_text().strip()

    result = json.loads(payload)
    gates = json.loads(gates_path.read_text())
    assert gates["schema"] == "tensorfem.verified-gates.v1"
    assert result["reference"]["kt"] == 3.0
    assert len(gates["gates"]) >= 8
    failed = [g["id"] for g in gates["gates"] if not g["passed"]]
    assert not failed, f"gates failed in committed evidence: {failed}"


def test_committed_gate_values_match_result_json():
    result = json.loads((BENCH / "solid3d_result.json").read_text())
    gates = {g["id"]: g for g in json.loads((BENCH / "solid3d_gates.json").read_text())["gates"]}
    finest = result["ladder"][-1]
    assert gates["peak_kt"]["value"] == pytest.approx(finest["peak"]["kt_estimate"], rel=1e-6)
    assert gates["pointwise_stress"]["value"] == pytest.approx(
        finest["pointwise_max_rel"], rel=1e-6
    )
    assert gates["szz_free_surfaces"]["value"] == pytest.approx(
        finest["field"]["szz_max_over_T"], rel=1e-6
    )
    assert gates["uz_plane_stress_kinematics"]["value"] == pytest.approx(
        finest["uz"]["uz_rel_l2"], rel=1e-6
    )
    kt = [r["kt_estimate"] for r in result["thickness_invariance"]]
    assert gates["thickness_invariance"]["value"] == pytest.approx(
        (max(kt) - min(kt)) / 3.0, rel=1e-6
    )


def test_committed_report_and_figures_exist():
    doc = ROOT / "docs" / "CASE-FEM-KIRSCH-PLATE-solid3d-verification.md"
    assert doc.exists()
    text = doc.read_text()
    assert "K_t = 3" in text or "K_t=3" in text
    assets = ROOT / "docs" / "assets" / "kirsch_plate_solid3d"
    pngs = sorted(p.name for p in assets.glob("*.png"))
    assert len(pngs) >= 4, pngs
    for name in pngs:
        assert (assets / name).stat().st_size > 5_000