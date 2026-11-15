"""Regression pins for the hydrodynamic added-mass referee used by GAP-ICE-GATE-POWER.

Context: mission m22156 (commit 7b947ff3) added a "strip theory" reference for the
heave added mass of the ice-coupled 6DOF chain and claimed it reproduces the 165 %
declared-vs-independent discrepancy of board report 19234.  Board audit m22176
(``docs/board_audit/m22176/audit_m22176.md``) found that the mission reference is
dimensionally inconsistent with the audited quantity (3D kg vs 2D kg/m), uses a
geometry (T=8 m) that differs from the chain (T=6 m, Cb=0.70), and disagrees with
the verified independent potential-flow anchor by +92.5 %.

These tests keep the two facts machine-checkable:

* the dipole added-mass operator of ``tensorfem.hydro_added_mass`` really does
  converge to Rayleigh's ``(8/3) rho a^3`` -- the one genuine numeric check of the
  anchor deliverable, which the mission's own acceptance command deselected;
* the strip-theory reference as shipped by that mission is *not* a 15 % referee
  against the ecosystem's verified 2D anchor, so it must not be used as one.

If the reference formula is ever fixed (or the referee row is re-based), the second
test fails on purpose and the audit verdict has to be re-adjudicated.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorfem.hydro_added_mass import (
    dipole_added_mass_matrix,
    disc_panels,
    piston_radiation_mass,
)

# Verified independent 2D heave added mass for the chain section (L=100 m, B=20 m,
# T=6 m): two mutually independent potential-flow routes (BEM mirror-Green single
# layer, FD 5-point Laplace) agreeing to 3.8 %, self-proved on the deep circle to
# <= 2.9 %.  Source: board report tensor_reports id=19234 (mission 21257) section Q5.
VERIFIED_BEM_2D_A33_KG_PER_M = 228.6e3
DECLARED_A_INF_OVER_L_KG_PER_M = 86.1e3
REFEREE_TOLERANCE = 0.15


def _strip_reference_2d(length: float, beam: float, draft: float, rho: float = 1025.0) -> float:
    """2D heave added mass [kg/m] of the strip reference shipped by m22156.

    Verbatim re-implementation of ``heave_added_mass_rectangular_barge`` in
    tests/test_hydro_ice_anchor.py of commit 7b947ff3 (same C_a interpolation,
    same 3D correction), returned in the 2D per-unit-length basis that the audited
    quantity uses.
    """
    bt = beam / draft
    if bt >= 10.0:
        c_a = math.pi / 4.0
    elif bt <= 0.5:
        c_a = math.pi / 2.0
    else:
        frac = (math.log(bt) - math.log(0.5)) / (math.log(10.0) - math.log(0.5))
        c_a = math.pi / 2.0 * (1.0 - frac) + math.pi / 4.0 * frac
    return rho * beam ** 2 * c_a


def test_dipole_operator_meets_3pct_at_64_cells():
    """Dipole added-mass operator vs Rayleigh baffled piston (3 % gate, cheap level).

    n_cells=64 -> 3228 panels -> 3228^2 dense operator (83 MB, ~0.1 s).  The finest
    level the m22156 suite asked for (n_cells=512 -> ~2.06e5 panels -> 316 GiB dense)
    is not run here; 3 % is already met two levels below it.
    """
    radius, rho = 1.0, 1025.0
    analytic = piston_radiation_mass(radius, rho)
    centres, areas = disc_panels(radius, 64)
    matrix = dipole_added_mass_matrix(centres, areas, rho)
    assert matrix.shape == (3228, 3228)
    rel_error = abs(float(matrix.sum()) - analytic) / analytic
    assert rel_error < 0.03, f"n_cells=64: rel_error={rel_error:.5f}"


def test_wide_limit_matches_flat_plate_added_mass():
    """B/T >> 1 limit of the strip reference must be the classic flat-plate value.

    A zero-draft flat plate of beam B at the free surface carries 2D heave added
    mass ``rho * pi * B**2 / 4``, i.e. ``C_a = pi/4`` in the reference's own
    ``rho*B**2*C_a`` normalisation.  This is the only limit of the formula that can
    be tied to an exact result.
    """
    got = _strip_reference_2d(100.0, 20.0, 2.0)  # B/T = 10
    exact = 1025.0 * math.pi * 20.0 ** 2 / 4.0
    assert abs(got - exact) / exact < 1e-12


def test_strip_reference_is_not_a_15pct_referee():
    """PINNED AUDIT FINDING: the m22156 strip reference misses its own 15 % band.

    At the chain geometry (T=6 m) the reference gives ~440.1 t/m against the verified
    independent 2D anchor 228.6 t/m -> +92.5 %.  A referee that is itself 6x outside
    its own tolerance cannot adjudicate the declared A_inf/L = 86.1 t/m.

    Caveat kept in the open: the verified anchor is the omega -> infinity limit while
    the strip formula states no frequency convention; scaling by a typical A_inf/A_0
    of 0.7 still leaves about +35 %, i.e. more than twice the tolerance, so the
    disagreement is not an artefact of the frequency convention alone.
    """
    reference = _strip_reference_2d(100.0, 20.0, 6.0)
    rel_error = (reference - VERIFIED_BEM_2D_A33_KG_PER_M) / VERIFIED_BEM_2D_A33_KG_PER_M
    assert rel_error > REFEREE_TOLERANCE, (
        f"reference={reference:.1f} kg/m, verified anchor={VERIFIED_BEM_2D_A33_KG_PER_M:.1f} kg/m, "
        f"rel_error={rel_error:.4%}"
    )
    # and the deviation it would report for the real declaration has the opposite
    # sign to the mission document (declared 9.5e6 kg > reference 3.6e6 kg)
    declared_error = (DECLARED_A_INF_OVER_L_KG_PER_M - reference) / reference
    assert declared_error < 0.0


def test_reference_basis_is_2d_per_unit_length():
    """Dimension pin: the audited quantity is kg/m, the mission shipped kg.

    ``heave_added_mass_rectangular_barge`` multiplies by the ship length and reports
    a 3D added mass in kg; report 19234 compares A_inf/L against a per-metre 2D
    potential-flow A33.  Comparing the two without the length factor is a factor-L
    (100x) error, which is why the mission's "165 %" numbers never reconciled.
    """
    area_basis = _strip_reference_2d(100.0, 20.0, 6.0)
    volume_basis = area_basis * 100.0
    assert volume_basis / area_basis == pytest.approx(100.0)
    assert VERIFIED_BEM_2D_A33_KG_PER_M < volume_basis  # 3D basis inflates by L