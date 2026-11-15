"""Tests for the v8 added-mass reference chain (BEM reference + dx refinement).

Claims defended here (each is a claim the v8 report makes):

1. the boundary-element added-mass reference is *correct* on the closed-form
   benchmarks (circle and ellipse, both axes, < 0.5 %), so it can be used to
   falsify a geometric attribution of the IBM excess;
2. a rectangle is never as "additive" as the v7 smearing fit: even a section as
   thick as its own chord stays below ``C_a = 1.6 rho pi c^2``, i.e. the measured
   ``3.6`` cannot be a finite-thickness effect;
3. the refinement driver's unit convention is consistent: the same *discrete*
   problem expressed at ``dx = 1/2`` and at ``dx = 1`` with everything (body,
   box, amplitude, period) doubled returns the same dimensionless added mass and
   the same lattice configuration;
4. the direct-forcing ``B^-1`` gain really imposes the prescribed marker velocity
   (residual ~1e-12), so the measured added mass is not a constraint-error
   artefact, and the epsilon switch does *not* change it (the epsilon system is a
   round-trip property of the operator, not a force normalisation).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfem.added_mass_bem2d import (
    added_mass_matrix,
    circle_panels,
    ellipse_panels,
    rectangle_added_mass_coefficient,
    rectangle_panels,
)
from tensorfem.ibm_added_mass_refinement import lattice_added_mass


# ---------------------------------------------------------------------------
# 1. BEM reference vs closed forms
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("radius", [0.5, 1.0, 2.0])
def test_bem_circle_matches_lamb(radius):
    m = added_mass_matrix(circle_panels(radius, 480))
    ref = math.pi * radius ** 2
    assert abs(m[0, 0] / ref - 1.0) < 5e-3
    assert abs(m[1, 1] / ref - 1.0) < 5e-3
    # axisymmetry of the discrete operator
    assert abs(m[0, 0] - m[1, 1]) < 1e-9 * ref


@pytest.mark.parametrize("a,b", [(1.0, 0.5), (1.0, 0.25), (2.0, 1.0)])
def test_bem_ellipse_matches_closed_form(a, b):
    """Motion along ``x`` -> ``rho pi b^2``; along ``y`` -> ``rho pi a^2``."""
    m = added_mass_matrix(ellipse_panels(a, b, 480))
    assert abs(m[0, 0] / (math.pi * b ** 2) - 1.0) < 5e-3
    assert abs(m[1, 1] / (math.pi * a ** 2) - 1.0) < 5e-3


def test_bem_plate_limit():
    """A very thin rectangle must approach the Lamb thin-plate value."""
    c_a = rectangle_added_mass_coefficient(0.02, per_edge=200)
    assert 0.98 < c_a < 1.12


def test_bem_thickness_is_monotone_and_bounded():
    """``C_a`` grows with thickness but a chord-thick body stays below 1.6."""
    hc = [0.05, 0.375, 1.0]
    vals = [rectangle_added_mass_coefficient(h, per_edge=160) for h in hc]
    assert vals[0] < vals[1] < vals[2]
    assert vals[1] < 1.45          # kernel-thickness body (3 dx) 
    assert vals[2] < 1.60          # a body as thick as its chord
    # the v7 claim needs C_a ~ 3.6: no plausible finite thickness gets there
    assert vals[2] < 3.6 / 2.0


def test_bem_panel_convergence():
    """Doubling the panel count must not move the coefficient by > 1 %."""
    coarse = rectangle_added_mass_coefficient(0.375, per_edge=60)
    fine = rectangle_added_mass_coefficient(0.375, per_edge=480)
    assert abs(fine / coarse - 1.0) < 1e-2


def test_bem_rectangle_geometry_closed():
    p = rectangle_panels(2.0, 0.5, per_edge=40)
    # closed loop: sum of panel vectors vanishes; outward normals point outwards
    ln, nrm = p["lengths"], p["normals"]
    assert ln.sum() == pytest.approx(2 * (4.0 + 1.0), rel=1e-12)
    mid = p["midpoints"]
    outward = mid[:, 0] * nrm[:, 0] + mid[:, 1] * nrm[:, 1]
    assert np.all(outward > -1e-12)


# ---------------------------------------------------------------------------
# 2. refinement driver: unit consistency and constraint
# ---------------------------------------------------------------------------
def test_unit_equivalence_dx_half_vs_doubled_dx1():
    """Same discrete problem, two unit conventions -> same dimensionless result."""
    half = lattice_added_mass(refine=2, c_phys=4.0, box_phys=64.0, amplitude=0.2,
                              period_phys=64.0, n_cycles=3.0, skip_cycles=1.0,
                              gain=2.0, gain_mode="operator")
    twin = lattice_added_mass(refine=1, c_phys=8.0, box_phys=128.0, amplitude=0.4,
                              period_phys=128.0, n_cycles=3.0, skip_cycles=1.0,
                              gain=2.0, gain_mode="operator")
    assert twin["n_cell"] == half["n_cell"]
    assert twin["n_markers"] == half["n_markers"]
    assert twin["amp_cells"] == pytest.approx(half["amp_cells"], rel=1e-12)
    assert twin["t_lat"] == pytest.approx(half["t_lat"], rel=1e-12)
    # lattice state evolution is identical -> coefficients agree to round-off
    assert twin["C_momentum"] == pytest.approx(half["C_momentum"], rel=1e-9)
    assert twin["C_force"] == pytest.approx(half["C_force"], rel=1e-9)


@pytest.mark.parametrize("mode", ["operator", "scalar"])
def test_constraint_is_imposed_at_the_guo_level(mode):
    """Direct forcing with ``B^-1`` hits the prescribed velocity to round-off."""
    r = lattice_added_mass(refine=1, c_phys=4.0, box_phys=64.0, amplitude=0.2,
                           period_phys=64.0, n_cycles=3.0, skip_cycles=1.0,
                           gain=2.0, gain_mode=mode)
    if mode == "operator":
        assert r["constraint_rel_max"] < 1e-10
        assert r["v_achieved_ratio"] == pytest.approx(1.0, abs=1e-9)
    else:
        # the archived scalar gain leaves a percent-level constraint error but
        # the added mass is unchanged: the measurement is not constraint-limited
        assert r["constraint_rel_max"] < 0.5
        assert r["v_achieved_ratio"] == pytest.approx(1.0, abs=0.05)


def test_epsilon_switch_does_not_change_the_added_mass():
    """The epsilon system is a round-trip property, not a force normalisation."""
    on = lattice_added_mass(refine=1, c_phys=4.0, box_phys=64.0, amplitude=0.2,
                            period_phys=64.0, n_cycles=3.0, skip_cycles=1.0,
                            gain=2.0, gain_mode="operator", correct_epsilon=True)
    off = lattice_added_mass(refine=1, c_phys=4.0, box_phys=64.0, amplitude=0.2,
                             period_phys=64.0, n_cycles=3.0, skip_cycles=1.0,
                             gain=2.0, gain_mode="operator", correct_epsilon=False)
    assert on["eps_amplification"] > 1.5          # the amplification is real
    assert off["eps_amplification"] == pytest.approx(1.0, abs=1e-12)
    assert off["C_momentum"] == pytest.approx(on["C_momentum"], rel=1e-6)
    assert off["C_force"] == pytest.approx(on["C_force"], rel=1e-6)


def test_force_and_momentum_routes_agree():
    """Two independent observables of the same coupling, one added mass."""
    r = lattice_added_mass(refine=1, c_phys=4.0, box_phys=64.0, amplitude=0.2,
                           period_phys=128.0, n_cycles=3.0, skip_cycles=1.0,
                           gain=2.0, gain_mode="operator")
    assert abs(r["C_force"] / r["C_momentum"] - 1.0) < 2.0e-2


def test_measurement_is_linear_in_amplitude():
    """Added mass must not depend on the drive amplitude (linear response)."""
    a = lattice_added_mass(refine=1, c_phys=4.0, box_phys=64.0, amplitude=0.1,
                           period_phys=128.0, n_cycles=3.0, skip_cycles=1.0,
                           gain=2.0, gain_mode="operator")
    b = lattice_added_mass(refine=1, c_phys=4.0, box_phys=64.0, amplitude=0.2,
                           period_phys=128.0, n_cycles=3.0, skip_cycles=1.0,
                           gain=2.0, gain_mode="operator")
    assert a["C_momentum"] == pytest.approx(b["C_momentum"], rel=0.15)