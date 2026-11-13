"""CASE-FEM-HERTZ-CONTACT: Hertz sphere-plane contact radius and indentation.

Acceptance: contact radius a and indentation delta within 3% of Hertz closed form
(Hertz 1882; Johnson 1985 §4.2) under prescribed load.
"""
import math

import numpy as np
import pytest

from tensorfem.hertz_halfspace_loadcontrol import (
    hertz_load_reference,
    solve_axisymmetric_contact_load_control,
    two_zone_axis,
    graded_depth,
)

# Problem parameters
RADIUS = 10.0
YOUNG = 1e5
POISSON = 0.3
LOAD = 1831.5


def _solve(h_over_a, depth_factor=96.0):
    """Solve at given h/a resolution."""
    ref = hertz_load_reference(LOAD, RADIUS, YOUNG, POISSON)
    a = ref.contact_radius
    n_inner = int(round(3.0 / h_over_a))
    r_axis = two_zone_axis(3.0 * a, depth_factor * a, n_inner, 22)
    z_axis = graded_depth(depth_factor * a, 4.0 * a, 20, 14)
    _, state, diag = solve_axisymmetric_contact_load_control(
        load=LOAD, radius=RADIUS, young=YOUNG, poisson=POISSON,
        r_axis=r_axis, z_axis=z_axis,
    )
    return ref, state, diag


def test_reference_oracle():
    """Hertz closed form self-consistency."""
    ref = hertz_load_reference(LOAD, RADIUS, YOUNG, POISSON)
    # a = (3PR/(4E*))^(1/3)
    e_star = YOUNG / (1 - POISSON**2)
    a_expected = (3.0 * LOAD * RADIUS / (4.0 * e_star)) ** (1.0 / 3.0)
    assert ref.contact_radius == pytest.approx(a_expected, rel=1e-12)
    # delta = a^2 / R
    assert ref.indentation == pytest.approx(ref.contact_radius**2 / RADIUS, rel=1e-12)
    # p0 = 3P / (2 pi a^2)
    assert ref.peak_pressure == pytest.approx(
        3.0 * LOAD / (2.0 * math.pi * ref.contact_radius**2), rel=1e-12
    )


def test_hertz_sphere_plane_a_and_delta_within_three_percent():
    """The case gate: a and delta within 3% at h/a=0.01."""
    ref, state, diag = _solve(h_over_a=0.01)
    assert diag["load_converged"]
    # Load balance: reaction matches prescribed load
    assert state.reaction == pytest.approx(ref.load, rel=1e-6)
    # Contact radius within 3%
    a_err = abs(state.contact_radius_linear / ref.contact_radius - 1.0)
    assert a_err < 0.03, f"a error = {100*a_err:.2f}% exceeds 3%"
    # Indentation within 3%
    delta_err = abs(state.prescribed_indentation / ref.indentation - 1.0)
    assert delta_err < 0.03, f"delta error = {100*delta_err:.2f}% exceeds 3%"
    # ALM penetration is negligible (no penalty bias)
    assert diag["penetration_over_indentation"] < 1e-4


def test_hertz_sphere_plane_a_converges_with_refinement():
    """Contact radius error decreases with mesh refinement."""
    _, s1, _ = _solve(h_over_a=0.05)
    _, s3, _ = _solve(h_over_a=0.01)
    ref = hertz_load_reference(LOAD, RADIUS, YOUNG, POISSON)
    e1 = abs(s1.contact_radius_linear / ref.contact_radius - 1.0)
    e3 = abs(s3.contact_radius_linear / ref.contact_radius - 1.0)
    # Finer mesh gives better contact radius
    assert e3 < e1, f"a error not improved: coarse={100*e1:.2f}%, fine={100*e3:.2f}%"


def test_alm_no_penalty_bias():
    """ALM penetration is at tolerance level, not penalty compliance."""
    _, state, diag = _solve(h_over_a=0.02)
    # Penetration / indentation < 1e-4 (ALM tolerance level)
    # vs penalty method which has ~2.9% systematic bias
    assert diag["penetration_over_indentation"] < 1e-4


def test_contact_noninvasive_outside_patch():
    """No contact outside the expected contact zone."""
    ref, state, _ = _solve(h_over_a=0.03)
    outside = state.surface_radii > 2.0 * ref.contact_radius
    max_p_outside = np.max(state.surface_pressure[outside])
    max_p_inside = np.max(state.surface_pressure)
    assert max_p_outside <= 1e-6 * max_p_inside
    # Gap is positive outside contact
    assert np.all(state.surface_gap[outside] > 0.0)
