"""Verification gates for :mod:`tensorfem.bimetal_thermoelastic`.

The module couples a thermal strain load to a Q9 plane-stress mesh and compares
the result with the Timoshenko (1925) closed form of a free bimetal strip.  The
tests below recompute every acceptance gate on meshes small enough for CI, plus
the consistency anchors that are cheap but sharp:

* the closed form is re-derived from a 2x2 closed-section equilibrium
  (independent of the formula's algebra);
* the layer-swap symmetry ``m -> 1/m, n -> 1/n, alpha1 <-> alpha2`` flips the
  sign of the curvature and keeps its magnitude;
* the single-material limit ``alpha1 = alpha2`` kills the curvature;
* the Q9 basis satisfies partition of unity and the nodal delta property, and
  every element of the strip mesh has a positive jacobian;
* the through-thickness refinement must not move the answer (the bending strain
  is linear in ``y`` and Q9 is exact for it);
* the stored evidence JSON must agree with a fresh solve.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tensorfem import bimetal_thermoelastic as bm

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "benchmarks" / "bimetal_thermoelastic_evidence.json"


# ---------------------------------------------------------------------------
# analytical side
# ---------------------------------------------------------------------------


def test_closed_form_matches_section_equilibrium():
    """The transcribed Timoshenko formula must equal the equilibrium derivation."""
    gap = bm.curvature_agreement()
    assert gap < 1e-10, f"closed form vs section equilibrium gap {gap:.3e}"


def test_unit_thermal_mismatch_form():
    """Default arguments reproduce the classical unit-mismatch expression."""
    m, n, dT, h = 1.0, 1.0, 100.0, 2.0e-3
    # m = n = 1 collapses the bracket to 3*4 + 2*2 = 16, so kappa = 6*dT*4/(16h)
    assert bm.timoshenko_curvature(m, n, dT, h) == pytest.approx(1.5 * dT / h, rel=1e-14)


@pytest.mark.parametrize("m,n", [(0.5, 0.5), (1.0, 1.0), (2.0, 3.0)])
def test_layer_swap_flips_curvature(m, n):
    """Exchanging the two layers must mirror the strip (kappa -> -kappa)."""
    case = bm.BimetalCase(m=m, n=n)
    swap = bm.BimetalCase(m=1.0 / m, n=1.0 / n, dT=case.dT, h=case.h,
                          length=case.length, E1=case.E2,
                          alpha1=case.alpha2, alpha2=case.alpha1, nu=case.nu)
    a = bm.analytical_solution(case)
    b = bm.analytical_solution(swap)
    assert b.curvature == pytest.approx(-a.curvature, rel=1e-12, abs=1e-18)
    assert b.tip_deflection == pytest.approx(-a.tip_deflection, rel=1e-12, abs=1e-18)


def test_curvature_linear_in_dT():
    """The closed form is linear in the temperature rise."""
    base = bm.analytical_solution(bm.BimetalCase(dT=100.0))
    for dT in (50.0, 200.0):
        sol = bm.analytical_solution(bm.BimetalCase(dT=dT))
        assert sol.curvature / base.curvature == pytest.approx(dT / 100.0, rel=1e-14)


def test_self_equilibrated_section():
    """Zero resultant force and zero resultant moment about the interface."""
    case = bm.BimetalCase(m=2.0, n=3.0)
    sol = bm.analytical_solution(case)
    ys = torch.linspace(-case.t2, case.t1, 20001, dtype=torch.float64)
    sig = torch.tensor([sol.sigma_xx(float(y)) for y in ys], dtype=torch.float64)
    dy = float(ys[1] - ys[0])
    force = float(sig.sum()) * dy                       # per unit width
    moment = float((sig * ys).sum()) * dy
    scale = case.E1 * abs(case.alpha1 - case.alpha2) * case.dT * case.h
    assert abs(force) < 1e-2 * scale
    assert abs(moment) < 1e-2 * scale * case.h


# ---------------------------------------------------------------------------
# element level
# ---------------------------------------------------------------------------


def test_q9_partition_of_unity_and_nodal_delta():
    coords = [(-1.0, -1.0), (-1.0, 0.0), (-1.0, 1.0),
              (0.0, -1.0), (0.0, 0.0), (0.0, 1.0),
              (1.0, -1.0), (1.0, 0.0), (1.0, 1.0)]
    for xi, eta in coords:
        n, dn_dxi, dn_deta = bm.q9_shape(xi, eta)
        assert float(n.sum()) == pytest.approx(1.0, rel=1e-15)
        exact = torch.tensor([1.0 if (abs(xi - x) < 1e-12 and abs(eta - y) < 1e-12) else 0.0
                              for x, y in coords], dtype=torch.float64)
        assert torch.allclose(n, exact, atol=1e-15)
        # derivatives of the basis must sum to zero (constant field)
        assert abs(float(dn_dxi.sum())) < 1e-15
        assert abs(float(dn_deta.sum())) < 1e-15


@pytest.mark.parametrize("m,n", [(0.5, 1.0), (1.0, 1.0), (2.0, 3.0)])
def test_element_jacobians_positive(m, n):
    case = bm.BimetalCase(m=m, n=n)
    mesh = bm.strip_mesh(case, 6, 3)
    _, detJ, _, _ = bm._gauss_data(case, mesh)          # raises if any det <= 0
    assert bool((detJ > 0).all())


def test_interface_node_line_is_shared():
    """Both layers share the y = 0 node row (bonded interface, no duplication)."""
    case = bm.BimetalCase(m=2.0, n=0.5)
    mesh = bm.strip_mesh(case, 4, 3)
    assert float(mesh.y_lines[3]) == pytest.approx(0.0, abs=1e-18)
    bottom = mesh.elements[mesh.element_index(0, 2)]
    top = mesh.elements[mesh.element_index(0, 3)]
    assert set(bottom.tolist()) & set(top.tolist()), "interface nodes are not shared"


# ---------------------------------------------------------------------------
# acceptance gates (campaign P189-N03)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def reference():
    case = bm.BimetalCase()
    res = bm.solve_bimetal_strip(case, 64, 4)
    return case, res, bm.analytical_solution(case)


def test_gate_a_tip_deflection(reference):
    """(a) free-end deflection, m = n = 1, dT = 100 K: < 3 %."""
    _, res, ref = reference
    err = bm.relative_error(res.tip_deflection, ref.tip_deflection)
    assert abs(err) < 0.03, f"tip deflection error {err:+.4%}"


def test_gate_a_sign_convention(reference):
    """The high-expansion layer on top makes the free end dip (kappa > 0 => w'' < 0)."""
    case, res, ref = reference
    assert ref.curvature > 0.0
    assert res.tip_deflection < 0.0
    assert res.curvature_estimate > 0.0


def test_gate_b_thermal_stress(reference):
    """(b) interface thermal stress in both layers: < 5 %."""
    _, res, ref = reference
    for value, target in ((res.sigma_interface_top, ref.sigma_interface_top),
                          (res.sigma_interface_bottom, ref.sigma_interface_bottom)):
        assert abs(bm.relative_error(value, target)) < 0.05


def test_gate_c_parameter_sweep():
    """(c) the 9 (m, n) combinations plus the 3-dT ladder: all errors < 5 %."""
    for row in bm.parameter_sweep(nx=32, ny_per_layer=4):
        assert abs(row["deflection_rel_error"]) < 0.05, row
        assert abs(row["sigma_top_rel_error"]) < 0.05, row
        assert abs(row["sigma_bottom_rel_error"]) < 0.05, row
        assert row["energy_identity_gap"] < 1e-6, row


def test_gate_d_convergence_order():
    """(d) observed order of the free-end deflection >= 1.8 (three-level triplet)."""
    study = bm.convergence_study(nxs=(8, 16, 32), ny_per_layer=4, grading=1.15)
    order = study["tip_deflection"]["order"]
    assert order == order and order >= 1.8, f"observed order {order}"
    gci = study["tip_deflection"]["gci"]
    assert 0.0 <= gci < 1e-2


def test_gate_d_uniform_family_order():
    """(d, cross-check) uniform family: order from the three finest levels >= 1.8."""
    study = bm.convergence_study(nxs=(32, 64, 128), ny_per_layer=4, grading=1.0)
    order = study["tip_deflection"]["order"]
    assert order == order and order >= 1.8, f"observed order {order}"


def test_gate_e_single_material_limit():
    """(e) alpha1 = alpha2: the curvature must vanish (< 1 % of the bimetal value)."""
    out = bm.single_material_limit()
    assert out["analytic_curvature"] == 0.0
    assert out["fem_relative_deflection"] < 0.01
    assert out["fem_curvature"] < 0.01 * bm.analytical_solution(bm.BimetalCase()).curvature


def test_gate_f_energy_balance():
    """(f) strain energy vs the closed form outside the clamp layer: < 2 %."""
    out = bm.energy_balance(nx=64, ny_per_layer=8)
    assert abs(out["region_rel_error"]) < 0.02, out
    assert out["energy_identity_gap"] < 1e-6, out
    # the whole-strip energy is *not* comparable: the closed form is invalid in
    # the clamped layer, which holds a large share of the energy.  The test pins
    # that documented behaviour so it cannot silently change sign/size.
    assert out["whole_strip_rel_error"] > 0.5
    assert out["region_fraction"] < 1.0


def test_section_equilibrium_of_the_fe_field(reference):
    """The FE stress field must be self-equilibrated (free end traction free)."""
    _, res, _ = reference
    case = res.case
    for frac in (0.5, 0.9):        # >= 2h from the singular free-end corner
        out = bm.section_resultants(res, x=frac * case.length)
        assert out["force_rel"] < 1e-6, out
        assert out["moment_rel"] < 1e-6, out


def test_thickness_direction_carries_no_error():
    """Refining n_y must not move the deflection (Q9 is exact for linear bending)."""
    rows = bm.thickness_independence(nx=32, nys=(2, 8))
    spread = abs(rows[0]["tip_rel_error"] - rows[1]["tip_rel_error"])
    assert spread < 1e-3, rows


def test_evidence_json_in_sync():
    """The committed evidence must match a fresh solve of the reference case."""
    data = json.loads(EVIDENCE.read_text())
    case = bm.BimetalCase()
    ref = bm.analytical_solution(case)
    mesh = data["meta"]["accuracy_mesh"]
    res = bm.solve_bimetal_strip(case, mesh["nx"], mesh["ny_per_layer"])
    rec = data["reference_case"]["fem"]
    assert res.tip_deflection == pytest.approx(rec["tip_deflection"], rel=1e-9)
    assert res.sigma_interface_top == pytest.approx(rec["sigma_interface_top"], rel=1e-9)
    assert res.sigma_interface_bottom == pytest.approx(rec["sigma_interface_bottom"], rel=1e-9)
    assert data["closed_form_check"]["max_rel_gap_vs_section_equilibrium"] < 1e-10
    assert data["reference_case"]["analytic"]["curvature"] == pytest.approx(ref.curvature,
                                                                           rel=1e-12)
    assert data["all_gates_pass"] is True
    assert all(g["pass"] for g in data["gates"])