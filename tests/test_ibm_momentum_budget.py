"""Regression tests for the IBM momentum budget / epsilon attribution (XVAL v7).

The claims these tests defend:

* the epsilon-conservation system is a *round-trip* property (``B 1 = 1``), it is
  not a force normalisation -- switching it off must not change the deposited
  per-marker force by more than the amplification factor, and the marker-side
  reaction must use exactly the weights the lattice receives;
* one LBM step deposits exactly the site-force sum in ``sum c f`` (the Guo source
  term is exact), while ``sum rho u`` lags it by the half-force split, so a
  constraint residual computed on ``sum c f / rho`` is *not* the physical
  velocity residual;
* the direct-forcing scheme with the archived gain 2 imposes the physical (Guo)
  velocity at the markers, and the ``B^-1`` (operator) gain does it to machine
  precision -- with a negligible change in the measured added mass;
* the measured apparent added mass of the discrete configuration is a property
  of the configuration: it is invariant to the epsilon switch (<= 0.5 %).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfem.ibm_aitken_added_mass import analytic_added_mass, plate_markers
from tensorfem.ibm_aitken_implicit import IBMInterface, LBMFluid2D
from tensorfem.ibm_momentum_budget import (
    constrained_velocity_residual,
    epsilon_amplification,
    ideal_direct_forcing_gain,
    lattice_momentum,
    one_step_force_budget,
    prescribed_plate_run,
    roundtrip_operator,
)


def _plate(c=4.0, box=64):
    return IBMInterface(plate_markers((0.5 * box, 0.5 * box), c), (box, box))


def test_epsilon_system_is_a_roundtrip_not_a_scale():
    """``eps`` makes ``interp o spread`` the identity on the uniform mode."""
    ibm = _plate()
    amp = epsilon_amplification(ibm)
    assert amp["roundtrip_error"] < 1e-12          # B 1 = 1, to round-off
    # ... while the *deposited* uniform force is amplified: the two numbers are
    # different objects and must not be confused.
    assert amp["amplification"] > 1.5
    assert math.isclose(amp["sum_eps_ds"], amp["amplification"] * amp["sum_ds"],
                        rel_tol=1e-12)


def test_spread_deposits_the_sum_the_marker_side_reports():
    """Action == reaction: the lattice receives ``sum(eps ds F)``, which is
    exactly the weight the marker-side reaction uses."""
    ibm = _plate()
    F = np.tile([0.0, 1.0], (len(ibm.positions), 1))
    site = ibm.spread(F)
    lattice_side = float(site[..., 1].sum())
    marker_side = float(np.dot(F[:, 1], ibm.eps_used * ibm.ds))
    assert math.isclose(lattice_side, marker_side, rel_tol=1e-12)


def test_epsilon_off_changes_the_force_scale_but_not_the_roundtrip_target():
    ibm_on = _plate()
    ibm_off = _plate()
    ibm_off.correct_epsilon = False
    on = epsilon_amplification(ibm_on)
    off = epsilon_amplification(ibm_off)
    assert math.isclose(off["amplification"], 1.0, rel_tol=1e-12)
    assert on["amplification"] > 1.5


def test_one_step_deposits_the_site_force_in_population_momentum():
    """``sum c f`` grows by exactly the site-force sum; ``sum rho u`` by half."""
    fluid = LBMFluid2D((16, 16))
    force = np.zeros((16, 16, 2))
    force[..., 1] = 0.01
    budget = one_step_force_budget(fluid, force)
    total = 256 * 0.01
    assert math.isclose(budget["population_delta"][1], total, rel_tol=1e-12)
    assert math.isclose(budget["guo_delta"][1], 0.5 * total, rel_tol=1e-9)


def test_roundtrip_operator_rows_sum_to_one():
    ibm = _plate()
    B = roundtrip_operator(ibm)
    assert np.allclose(B.sum(axis=1), 1.0, atol=1e-12)
    assert np.linalg.cond(B) < 1e3


def test_guo_velocity_constraint_is_imposed_and_raw_velocity_is_not():
    """The archived scalar gain imposes the *physical* (Guo) velocity.

    The raw ``sum c f / rho`` residual is much larger: reporting it as the
    constraint error is a measurement error, not a scheme defect.
    """
    rec = prescribed_plate_run(n_steps=200, gain=2.0)
    assert rec["residual_guo_max"] < 0.25
    assert rec["residual_raw_max"] > 3.0 * rec["residual_guo_max"]


def test_operator_gain_imposes_the_constraint_exactly():
    rec = prescribed_plate_run(n_steps=200, gain=2.0, gain_mode="operator")
    assert rec["residual_guo_max"] < 1e-10


def test_added_mass_is_invariant_to_the_epsilon_switch_when_constraint_is_exact():
    """With the constraint imposed exactly, epsilon cannot reach the added mass.

    In the ``operator`` (``B^-1``) gain the marker velocity is pinned to
    ``u_body`` to round-off, so the flow -- and hence the force -- is fixed by
    the Dirichlet condition alone; the epsilon system then only decides how much
    Lagrange force buys that velocity.  This is the controlled experiment that
    refutes the "divide the added mass by the epsilon amplification" reading of
    ``docs/xval_implicit_ibm_aitken_v6.md`` §3.2.
    """
    on = prescribed_plate_run(n_steps=200, gain=2.0, gain_mode="operator",
                              correct_epsilon=True)
    off = prescribed_plate_run(n_steps=200, gain=2.0, gain_mode="operator",
                               correct_epsilon=False)
    assert math.isclose(on["m_force"], off["m_force"], rel_tol=1e-9)


def test_scalar_gain_couples_its_own_effective_value_to_epsilon():
    """Documented defect of the *scalar* gain: it is an epsilon-scaled gain.

    Dropping the epsilon correction scales the effective direct-forcing gain by
    ``1/amplification``, so the archived added mass moves by tens of percent
    even though the physics (the Dirichlet condition) is untouched.  A scalar
    gain therefore cannot be quoted as gain-independent evidence.
    """
    on = prescribed_plate_run(n_steps=200, gain=2.0, correct_epsilon=True)
    off = prescribed_plate_run(n_steps=200, gain=2.0, correct_epsilon=False)
    assert abs(on["m_force"] / off["m_force"] - 1.0) > 0.2


def test_force_and_momentum_routes_agree():
    """The reported interface force equals the momentum actually deposited, so
    the 'momentum route' cannot be used to 'correct' the force route."""
    rec = prescribed_plate_run(n_steps=200, gain=2.0)
    assert abs(rec["m_force"] / rec["m_momentum"] - 1.0) < 0.05


def test_constraint_residual_reports_both_components():
    ibm = _plate()
    rho = np.ones((64, 64))
    u = np.zeros((64, 64, 2))
    res = constrained_velocity_residual(ibm, rho, u, np.array([0.0, 0.01]))
    assert res["max_rel"] > 0.9          # nothing is moving yet
    assert len(res["max_err"]) == 2


def test_ideal_gain_is_the_guo_half_split_inverse():
    assert math.isclose(ideal_direct_forcing_gain(1.5), 2.0 / 3.0, rel_tol=1e-12)


def test_reference_added_mass_is_lamb():
    assert math.isclose(analytic_added_mass(4.0), math.pi * 16.0, rel_tol=1e-12)