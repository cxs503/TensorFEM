"""Regression guard: the load-controlled Hertz solver must audit its own equilibrium.

The free-dof residual ``(K u)_free + lambda`` is reported as
``metadata["equilibrium_residual_rel"]``.  Before the fix in this commit the
external-force vector was assembled with the wrong sign (``f_ext = -lambda``),
so the audit reported exactly ``2.0`` on a solution whose reaction matches the
prescribed load to 1e-14 -- i.e. the quantity could never pass, and a genuine
equilibrium violation would have been hidden behind the constant offset.
"""
from __future__ import annotations

import numpy as np

from tensorfem.hertz_halfspace_loadcontrol import (
    graded_depth,
    hertz_load_reference,
    solve_axisymmetric_contact_load_control,
    two_zone_axis,
)


def _solve(nr=32, nz=24):
    radius, young, nu = 10.0, 1.0e5, 0.3
    # a/R = 0.05 Hertz load for the rigid-indenter reduction.
    e_star = young / (1.0 - nu**2)
    a = 0.05 * radius
    load = 4.0 * e_star * a**3 / (3.0 * radius)
    ref = hertz_load_reference(load, radius, young, nu)
    a = ref.contact_radius
    r_axis = two_zone_axis(3.0 * a, 24.0 * a, nr, nz)
    z_axis = graded_depth(24.0 * a, 4.0 * a, 12, 8)
    return solve_axisymmetric_contact_load_control(
        load=load, radius=radius, young=young, poisson=nu,
        r_axis=r_axis, z_axis=z_axis,
        load_tol=1e-10, keep_displacement=True,
    )


def test_equilibrium_residual_is_small_not_two():
    pre, state, diag = _solve()
    rel = state.metadata["equilibrium_residual_rel"]
    assert np.isfinite(rel)
    # the historical defect produced 2.0 exactly; anything near it means the
    # sign convention regressed.
    assert rel < 1e-4, rel
    assert abs(rel - 2.0) > 1.0, rel


def test_global_force_balance_matches_prescribed_load():
    """The ALM load-control loop must land on the prescribed normal force.

    Tolerance is loose compared with the anchor certificate (which grades the
    converged fine mesh): this coarse mesh is a fast regression check, and the
    accepted ALM iterate is the least-penetrating one, not necessarily the last.
    """
    pre, state, diag = _solve()
    assert abs(state.reaction / pre.load - 1.0) < 1e-6
    assert abs(float(np.sum(state.multiplier)) / pre.load - 1.0) < 1e-6
