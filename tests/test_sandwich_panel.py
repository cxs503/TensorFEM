"""Verification tests for the sandwich-panel bending module.

The chain of independent statements:

* the generalised (heterogeneous) laminate is a *strict superset* of the old
  equal-thickness/single-material laminate -- identical ABD, identical
  interfaces, identical 3-D elasticity answer (regression, no silent change);
* the sandwich geometry reproduces the classical rigidities by hand
  (``D = Q_face t_f d^2 / 2``, ``D_Q = G_core d^2 / t_core``);
* homogeneous limit: a panel whose core is the face material must reproduce the
  monolithic-plate CLT answer (so the sandwich code cannot invent stiffness);
* the antiplane-core closed form and the sandwich-corrected FSDT agree with the
  layered 3-D elasticity solution to <3 % for engineering spans -- the accuracy
  claim of the module;
* the uniform shear correction ``k = 5/6`` (what a homogeneous-plate code
  applies) is *not* usable for a sandwich: its error is >50 % at a/h = 18.  This
  is asserted so the regression cannot silently come back;
* face stress and core shear match the 3-D solution, and the HEX20 path
  reproduces the same panel answer (independent of the state-space solver).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfem.composite_laminate_bending import (
    PAGANO_MATERIAL,
    Laminate,
    LayeredExact3D,
    layered_hex20_fem,
    navier_clt,
    ply_stiffness_3d,
)
from tensorfem.sandwich_panel import (
    PVC_CORE,
    ELASTIC_FACE,
    SandwichPanel,
    accuracy_envelope,
    deflection_budget,
)


# ── the generalised laminate is backwards compatible ────────────────────────
def test_per_ply_overrides_reproduce_the_uniform_laminate():
    uniform = Laminate([0, 90, 90, 0], 2.5e-3, PAGANO_MATERIAL)
    general = Laminate([0, 90, 90, 0], 2.5e-3, PAGANO_MATERIAL,
                       ply_materials=[PAGANO_MATERIAL] * 4,
                       ply_thicknesses=[2.5e-3] * 4)
    for (A1, B1, D1, As1), (A2, B2, D2, As2) in ((uniform.abd(), general.abd()),):
        assert np.allclose(A1, A2, rtol=0, atol=0)
        assert np.allclose(B1, B2, rtol=0, atol=0)
        assert np.allclose(D1, D2, rtol=0, atol=0)
        assert np.allclose(As1, As2, rtol=0, atol=0)
    assert uniform.z_interfaces == general.z_interfaces
    assert uniform.thickness == general.thickness
    e1 = LayeredExact3D(uniform, 1.0, 1.0).center_deflection
    e2 = LayeredExact3D(general, 1.0, 1.0).center_deflection
    assert e1 == e2


def test_heterogeneous_inputs_are_validated():
    with pytest.raises(ValueError, match="one entry per ply"):
        Laminate([0, 90], 1e-3, PAGANO_MATERIAL,
                 ply_materials=[PAGANO_MATERIAL])
    with pytest.raises(ValueError, match="one entry per ply"):
        Laminate([0, 90], 1e-3, PAGANO_MATERIAL, ply_thicknesses=[1e-3])
    with pytest.raises(ValueError, match="positive"):
        Laminate([0, 90], 1e-3, PAGANO_MATERIAL, ply_thicknesses=[1e-3, 0.0])
    with pytest.raises(TypeError):
        Laminate([0], 1e-3, PAGANO_MATERIAL, ply_materials=[1.0])
    with pytest.raises(ValueError):
        SandwichPanel(t_core=0.0)
    with pytest.raises(ValueError):
        SandwichPanel(n_core=0)


# ── geometry and classical rigidities ──────────────────────────────────────
def test_sandwich_geometry_and_rigidities_match_hand_formulas():
    p = SandwichPanel(face=ELASTIC_FACE, core=PVC_CORE, t_face=2e-3, t_core=40e-3)
    assert math.isclose(p.thickness, 44e-3)
    assert math.isclose(p.d, 42e-3)
    lam = p.laminate()
    A, B, D, As = lam.abd()
    Q = p.face_Q()
    # symmetric: no extension-bending coupling, faces contribute the membrane terms
    assert np.max(np.abs(B)) < 1e-12 * np.max(np.abs(D))
    t_f, t_c = p.t_face, p.t_core
    assert math.isclose(A[0, 0], 2 * Q[0, 0] * t_f + 90e6 * t_c / (1 - 0.4 ** 2),
                        rel_tol=1e-9)
    # the antiplane-core bending rigidity is the face-dominant part of the full D
    Dc = p.classic_D()
    assert Dc[0, 0] < D[0, 0] < 1.2 * Dc[0, 0]
    # Allen/Plantema shear rigidity, recovered from the core modulus only
    assert math.isclose(p.classic_shear_rigidity(),
                        p.core_shear_modulus() * p.d ** 2 / p.t_core, rel_tol=1e-12)
    assert math.isclose(p.core_only_shear_rigidity(),
                        p.core_shear_modulus() * t_c, rel_tol=1e-12)
    # ratio that drives the error of a uniform k=5/6 in a sandwich
    assert p.classic_shear_rigidity() / p.core_only_shear_rigidity() > 1.05


def test_homogeneous_limit_reproduces_the_monolithic_plate():
    # core == face and equal thicknesses: the panel is a plain plate
    p = SandwichPanel(face=ELASTIC_FACE, core=ELASTIC_FACE, t_face=5e-3, t_core=10e-3,
                      n_core=1)
    mono = Laminate([0.0], 20e-3, ELASTIC_FACE)
    a = 0.5
    assert math.isclose(p.clt(a).W, navier_clt(mono, a, a).W, rel_tol=1e-12)
    assert math.isclose(p.laminate().thickness, 20e-3)


# ── accuracy claim: antiplane-core theory vs the layered 3-D solution ───────
@pytest.mark.parametrize("a_over_h", [9.0, 18.0, 40.0])
def test_antiplane_and_sandwich_fsdt_match_the_3d_solution(a_over_h):
    p = SandwichPanel()
    row = deflection_budget(p, a_over_h * p.thickness)
    assert abs(row["err_classical"]) < 0.03, row
    assert abs(row["err_fsdt_sandwich"]) < 0.03, row
    # face stress (top face, tension positive) and core shear magnitude
    assert abs(row["err_sigma_face_classical"]) < (0.10 if a_over_h < 12 else 0.05), row
    tau_ratio = row["classical_tau_core_center"] / row["exact3d_tau_core_center"]
    assert abs(abs(tau_ratio) - 1.0) < 0.03, row
    # opposite sign convention (engineering Q/d vs the z-up 3-D convention)
    assert tau_ratio < 0.0


def test_uniform_shear_correction_is_rejected_for_sandwich():
    """the homogeneous-plate answer (CLT and FSDT k=5/6) is >50 % too stiff."""
    p = SandwichPanel()
    row = deflection_budget(p, 18.0 * p.thickness)
    assert row["err_clt"] < -0.50, row
    assert row["err_fsdt_k56"] < -0.50, row
    # the core-only (k = 1) rigidity is already within 10 %, the d^2/t_c one <3 %
    assert abs(row["err_fsdt_core"]) < 0.10, row
    assert abs(row["err_fsdt_sandwich"]) < 0.03, row


def test_deflection_budget_flags_the_thick_panel_limit():
    """At a/h ~ 4.5 the antiplane core model is outside 3 % -- documented, not hidden."""
    p = SandwichPanel()
    rows = accuracy_envelope(p, spans=[4.5 * p.thickness, 9.0 * p.thickness])
    assert abs(rows[0]["err_classical"]) > 0.01
    assert abs(rows[1]["err_classical"]) < 0.03


def test_stiffer_core_reduces_the_shear_share():
    soft = SandwichPanel(core=PVC_CORE)
    stiff = SandwichPanel(core=PVC_CORE.__class__(**{**PVC_CORE.__dict__,
                                                     "G12": 320e6, "G13": 320e6, "G23": 320e6,
                                                     "E1": 900e6, "E2": 900e6, "E3": 900e6}))
    a = 18.0 * soft.thickness
    cs = soft.classical(a)
    ct = stiff.classical(a)
    assert ct["w_center"] < cs["w_center"]
    assert ct["w_shear"] / ct["w_center"] < cs["w_shear"] / cs["w_center"]


@pytest.mark.slow
def test_hex20_reproduces_the_sandwich_3d_solution():
    p = SandwichPanel(n_core=2)
    a = 9.0 * p.thickness
    ex = float(p.exact3d(a).center_deflection)
    fe = p.hex20(a, nx=4, nz_per_ply=1)
    assert abs(fe["w_center"] / ex - 1.0) < 0.02, (fe["w_center"], ex)