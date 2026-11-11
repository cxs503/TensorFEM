"""Exact Saint-Venant end traction and wall-peak recovery for rectangular torsion.

Companion of ``docs/torsion-rectangular-end-traction.md``.  The existing
``tests/test_torsion_rectangular_section.py`` keeps the *fail-closed* record of the
original load case (``end_traction="linear"``, where the a/b = 3 twist rate drifts away
under refinement); these tests pin the cure and its attribution, and the wall-peak
recovery that §5.2 of the rectangular report left open.
"""
from __future__ import annotations

import pytest
import torch

from tensorfem.torsion_bar import rectangular_torsion_constant
from tensorfem.torsion_rectangular import (
    _twist_prism,
    rectangular_torsion_bar,
    shape_factors,
    surface_peak_shear,
)


def test_exact_traction_is_a_pure_torque_and_reproduces_the_closed_form_rate():
    """The Gauss-integrated Saint-Venant traction must carry exactly ``T``, no net force."""
    result = rectangular_torsion_bar(a=3.0, b=1.0, length=4.0, segments_short=3,
                                     end_traction="exact")
    assert result.applied_torque == pytest.approx(1.0, abs=1e-12)
    # quadrature of the exact traction reproduces the torque before rescaling
    assert result.end_traction_residual < 5e-3
    assert result.end_traction == "exact"


def test_unknown_end_traction_is_rejected():
    with pytest.raises(ValueError, match="end_traction"):
        rectangular_torsion_bar(a=2.0, b=1.0, length=4.0, segments_short=2,
                                end_traction="saint-venant")


def test_slender_section_converges_once_the_end_disturbance_is_removed():
    """a/b = 3 at L = 4 (a/L = 0.75) is the configuration recorded as divergent.

    With the *physical* linear traction the error grows under refinement (see the
    fail-closed test in ``test_torsion_rectangular_section.py``).  With the exact
    traction the same prism converges monotonically and crosses the 3 % gate, which is
    what identifies the end zone -- not the section shape -- as the cause.
    """
    coarse = rectangular_torsion_bar(a=3.0, b=1.0, length=4.0, segments_short=3,
                                     end_traction="exact")
    fine = rectangular_torsion_bar(a=3.0, b=1.0, length=4.0, segments_short=4,
                                   end_traction="exact")
    assert fine.torsion_constant_relative_error < coarse.torsion_constant_relative_error
    assert fine.torsion_constant_relative_error < 0.03
    # the two load cases cannot be the same experiment
    linear = rectangular_torsion_bar(a=3.0, b=1.0, length=4.0, segments_short=4,
                                     end_traction="linear")
    assert linear.torsion_constant_relative_error > 1.5 * fine.torsion_constant_relative_error


def test_exact_traction_makes_the_twist_rate_window_independent():
    """The measuring window stops mattering once the end traction is the exact one."""
    windows = ((0.25, 0.75), (0.35, 0.65))
    rates = {}
    for traction in ("linear", "exact"):
        solve = _twist_prism(a=3.0, b=1.0, length=4.0, segments_short=3, segments_long=None,
                             young=1.0e9, poisson=0.3, torque=1.0, window=windows[0],
                             terms=401, rtol=1e-12, end_traction=traction)
        measured = [solve.twist_rate_in(window) for window in windows]
        exact = 1.0 / ((1.0e9 / 2.6) * rectangular_torsion_constant(3.0, 1.0))
        rates[traction] = [abs(value / exact - 1.0) for value in measured]
    linear_spread = abs(rates["linear"][0] - rates["linear"][1])
    exact_spread = abs(rates["exact"][0] - rates["exact"][1])
    assert linear_spread > 3e-3
    assert exact_spread < 0.2 * linear_spread


def test_wall_peak_is_larger_on_the_long_side_and_is_recovered():
    """Wall shear at both mid-side points; the peak must sit on the long side."""
    peak = surface_peak_shear(a=2.0, b=1.0, length=6.0, segments_short=4)
    assert peak.long_over_short_exact == pytest.approx(
        shape_factors(2.0, 1.0).long_over_short, rel=1e-12)
    assert peak.long_over_short_exact > 1.2
    assert peak.direct_long > peak.direct_short
    # the direct (wall-node) recovery is the usable scheme at 4 % on this mesh
    assert peak.relative_error("direct") < 0.04
    assert peak.relative_error("direct", "short") < 0.15
    # the half-element-inside centroid value is a strict interior lower bound
    assert peak.centroid_long < 0.75 * peak.exact_long
    assert peak.gauss_long == pytest.approx(peak.direct_long, rel=0.05)


def test_surface_peak_needs_a_node_on_the_symmetry_plane():
    with pytest.raises(ValueError, match="even"):
        surface_peak_shear(a=1.5, b=1.0, length=6.0, segments_short=3)


def test_twist_prism_rotation_profile_is_linear_in_the_middle():
    """Saint-Venant rotation is linear in z away from the ends: R^2 of the fit ~ 1."""
    solve = _twist_prism(a=2.0, b=1.0, length=4.0, segments_short=2, segments_long=None,
                         young=1.0e9, poisson=0.3, torque=1.0, window=(0.25, 0.75),
                         terms=401, rtol=1e-12, end_traction="exact")
    inside = (solve.planes >= 0.25 * solve.length) & (solve.planes <= 0.75 * solve.length)
    slope = solve.twist_rate_in((0.25, 0.75))
    residual = solve.thetas[inside] - (solve.planes[inside] * slope + solve.thetas[inside][0]
                                       - solve.planes[inside][0] * slope)
    assert float(torch.sqrt(torch.mean(residual**2))) < 1e-3 * abs(slope) * solve.length