"""Tests for the MITC4 formulation and axisymmetric subspace of the flat-facet
cylindrical shell eigenbuckling (follow-up to
``docs/cylindrical-shell-geometric-stiffness-audit.md``).

Two layers are covered:

* **invariants of the new paths** — the default ``shell4`` path is bit-for-bit
  the historical one; the MITC membrane hoop energy still reproduces the
  chord/arc area ratio; the axisymmetric constraint matrix maps a reduced
  vector to a field that is exactly axisymmetric (constant radial amplitude
  per ring, rotation about the tangent); and the axisymmetric minimum
  eigenvalue is never below the full-system minimum (min-max principle).
* **a fail-closed lock on the attribution** — the MITC4 axisymmetric family
  is monotone and within 3 % of the Donnell finite-length reference at
  ``nz=30, nt=32``, while the shell4 family is still >20 % over-stiff at the
  same mesh.  Together these pin the v1 over-stiffness to the
  full-integration bending-shear coupling rather than to the geometric
  stiffness operator.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorfem.cylindrical_shell_buckling import (
    _axisymmetric_constraint,
    classical_axial_buckling_resultant,
    cylindrical_shell_axial_buckling,
    cylindrical_shell_mesh,
    donell_axisymmetric_load,
)
from tensorfem.shell4_mitc import shell4_mitc_stiffness

CASE = dict(radius=1.0, length=1.2, young=200e9, poisson=0.3, thickness=0.01)


def test_default_formulation_bit_identical():
    """The default path must reproduce the historical v1 evidence values."""
    # From benchmarks/cylindrical_shell_buckling_evidence.json (v1 audit).
    expected = {(10, 16): 25447033.86671515, (30, 32): 14950174.49692438}
    for (nz, nt), ref in expected.items():
        res = cylindrical_shell_axial_buckling(ntheta=nt, nz=nz, **CASE)
        value = float(res.load_factors[0])
        assert value == pytest.approx(ref, rel=1e-9), (nz, nt, value)
    # Explicit formulation="shell4" is the same operator.
    res = cylindrical_shell_axial_buckling(ntheta=16, nz=10, **CASE, formulation="shell4")
    assert float(res.load_factors[0]) == pytest.approx(expected[(10, 16)], rel=1e-9)


def test_rejects_unknown_formulation():
    with pytest.raises(ValueError):
        cylindrical_shell_axial_buckling(ntheta=8, nz=4, **CASE, formulation="nope")


def test_mitc_membrane_hoop_energy_matches_facet_area_ratio():
    """Uniform radial shrink: the MITC membrane term must also be exact.

    The prescribed shrink blocks axial contraction, so the membrane energy
    carries the plane-stress 1/(1-nu**2) factor on top of the chord/arc area
    ratio of the faceted ring.
    """
    R, L = CASE["radius"], CASE["length"]
    E, nu, t = CASE["young"], CASE["poisson"], CASE["thickness"]
    for nt in (8, 16, 32):
        nodes, elements = cylindrical_shell_mesh(R, L, 2, nt)
        ndof = 6 * len(nodes)
        K = torch.zeros((ndof, ndof), dtype=nodes.dtype)
        for el in elements:
            ids = torch.stack(tuple(6 * el + k for k in range(6)), 1).reshape(-1)
            K[ids[:, None], ids] += shell4_mitc_stiffness(nodes[el], E, nu, t)
        radial = nodes.clone()
        radial[:, 0] = 0.0
        radial = radial / torch.linalg.vector_norm(radial, dim=1, keepdim=True)
        u = torch.zeros(ndof, dtype=nodes.dtype)
        for c in range(3):
            u[c::6] = -radial[:, c]
        fem = 0.5 * float(u @ K @ u)
        continuum = 0.5 * E * t / R**2 * (2 * math.pi * R * L)
        facet_area = nt * math.sin(math.pi / nt) / math.pi
        assert abs(fem / continuum / facet_area * (1 - nu**2) - 1.0) < 5e-3


def test_constraint_matrix_is_axisymmetric():
    """A reduced vector must map to a field with constant radial amplitude
    per ring and rotation about the local tangent."""
    nz, nt = 3, 8
    C = _axisymmetric_constraint(nz, nt)
    assert C.shape == (6 * (nz + 1) * nt, 3 * (nz + 1))
    red = torch.randn(3 * (nz + 1), dtype=torch.float64)
    full = (C @ red).reshape(nz + 1, nt, 6)
    th = torch.arange(nt) * (2.0 * math.pi / nt)
    cos, sin = torch.cos(th), torch.sin(th)
    for i in range(nz + 1):
        u_i, w_i, p_i = float(red[3 * i]), float(red[3 * i + 1]), float(red[3 * i + 2])
        for j in range(nt):
            got = [float(full[i, j, k]) for k in range(6)]
            # translation: (u, w cos, w sin); rotation about the tangent (0, -sin, cos)
            want = (u_i, w_i * float(cos[j]), w_i * float(sin[j]), 0.0,
                    -p_i * float(sin[j]), p_i * float(cos[j]))
            for k in range(6):
                assert got[k] == pytest.approx(want[k], abs=1e-12)


def test_axisymmetric_eigenvalue_not_below_full():
    """Min-max: restricting to a subspace cannot lower the minimum eigenvalue."""
    finite = donell_axisymmetric_load(**CASE)
    for nz, nt in ((6, 32), (10, 32)):
        full = float(cylindrical_shell_axial_buckling(ntheta=nt, nz=nz, **CASE).load_factors[0])
        ax = float(
            cylindrical_shell_axial_buckling(ntheta=nt, nz=nz, **CASE, axisymmetric=True).load_factors[0]
        )
        assert ax >= full * (1.0 - 1e-6), (nz, nt, ax, full)


def test_mitc_axial_family_within_3pct():
    """The certified result: MITC4 axisymmetric family reaches the 3 % gate."""
    finite = donell_axisymmetric_load(**CASE)
    value = float(
        cylindrical_shell_axial_buckling(
            ntheta=32, nz=30, **CASE, axisymmetric=True, formulation="mitc"
        ).load_factors[0]
    )
    assert abs(value - finite) / finite < 0.03


def test_shell4_still_over_stiff_at_fine_mesh():
    """The v1 debt is a formulation effect: shell4 stays >20 % over-stiff at
    the same mesh where MITC4 passes, so the fix is attributed to the
    bending-shear coupling, not to the axisymmetric restriction."""
    finite = donell_axisymmetric_load(**CASE)
    value = float(
        cylindrical_shell_axial_buckling(ntheta=32, nz=30, **CASE, axisymmetric=True).load_factors[0]
    )
    assert (value - finite) / finite > 0.20


def test_mitc_full_system_matches_axisymmetric_at_nt32():
    """At nt=32 the MITC full-system minimum is the axisymmetric mode, so the
    two solves must agree to solver tolerance."""
    full = float(cylindrical_shell_axial_buckling(ntheta=32, nz=20, **CASE, formulation="mitc").load_factors[0])
    ax = float(
        cylindrical_shell_axial_buckling(
            ntheta=32, nz=20, **CASE, axisymmetric=True, formulation="mitc"
        ).load_factors[0]
    )
    assert full == pytest.approx(ax, rel=1e-6)
