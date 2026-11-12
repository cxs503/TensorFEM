"""Acceptance tests for the load-controlled Hertz contact face.

Case FEM-HERTZ-CONTACT: contact radius ``a`` and indentation ``delta`` under a
prescribed load, against the Hertz closed form, tolerance 3% (case pool).
"""
import math

import numpy as np
import pytest

from tensorfem.hertz_halfspace_loadcontrol import (
    axisymmetric_mesh,
    graded_depth,
    hertz_load_reference,
    solve_axisymmetric_contact_load_control,
    surface_ring_areas,
    two_zone_axis,
)

R, E, NU = 10.0, 1.0e5, 0.3


def case_setup(a_over_r=0.05, h_over_a=0.015, depth_factor=48.0):
    ref = hertz_load_reference(
        4.0 * (E / (1 - NU**2)) * (a_over_r * R) ** 3 / (3.0 * R), R, E, NU
    )
    a = ref.contact_radius
    r_axis = two_zone_axis(3 * a, depth_factor * a, int(round(3 / h_over_a)), 22)
    z_axis = graded_depth(depth_factor * a, 4 * a, 20, 14)
    return ref, r_axis, z_axis


def solve(ref, r_axis, z_axis, **kw):
    return solve_axisymmetric_contact_load_control(
        load=ref.load, radius=R, young=E, poisson=NU,
        r_axis=r_axis, z_axis=z_axis, **kw
    )


def test_reference_solution_inverts_itself():
    ref = hertz_load_reference(1831.5, 10.0, 1e5, 0.3)
    assert ref.contact_radius == pytest.approx((3 * ref.load * ref.radius
                                                / (4 * ref.effective_modulus)) ** (1 / 3))
    assert ref.indentation == pytest.approx(ref.contact_radius**2 / ref.radius)
    assert ref.peak_pressure == pytest.approx(3 * ref.load / (2 * math.pi * ref.contact_radius**2))
    # the pressure field integrates to the applied load:
    # P = int_0^a p(r) 2 pi r dr
    r = np.linspace(0.0, ref.contact_radius, 20001)
    load_from_field = np.trapezoid(ref.pressure(r) * 2.0 * np.pi * r, r)
    assert load_from_field == pytest.approx(ref.load, rel=1e-3)


def test_reference_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        hertz_load_reference(0.0, 10.0, 1e5, 0.3)
    with pytest.raises(ValueError):
        hertz_load_reference(1.0, 10.0, 1e5, 0.7)


def test_ring_areas_reproduce_the_surface_area():
    radii = axisymmetric_mesh(20, 4, 5.0, 2.0)[0][np.arange(0, 21 * 5, 5), 0]
    areas = surface_ring_areas(radii)
    assert np.all(areas > 0)
    assert areas.sum() == pytest.approx(math.pi * radii[-1] ** 2, rel=1e-14)


def test_stiffness_assembly_matches_reference_implementation():
    import torch
    from tensorfem.hertz_axisymmetric import (
        assemble_axisymmetric_stiffness as ref_assembly,
        axisymmetric_mesh as ref_mesh,
    )
    from tensorfem.hertz_halfspace_loadcontrol import assemble_axisymmetric_stiffness

    nodes, quads = axisymmetric_mesh(5, 4, 6.0, 12.0)
    rnodes, relems = ref_mesh(5, 4, 6.0, 12.0, dtype=torch.float64)
    mine = assemble_axisymmetric_stiffness(nodes, quads, 1e5, 0.3).toarray()
    ref = ref_assembly(rnodes, relems, 1e5, 0.3).numpy()
    assert np.allclose(nodes, rnodes.numpy())
    assert np.abs(mine - ref).max() < 1e-6 * np.abs(ref).max()


def test_load_controlled_hertz_within_three_percent():
    """The case gate itself: a and delta within 3% of Hertz at a fixed load."""
    # depth 96a: the finite-patch truncation (see the truncation study) must be
    # inside the 3% budget before the mesh error is judged
    ref, r_axis, z_axis = case_setup(depth_factor=96.0)
    ref, state, diag = solve(ref, r_axis, z_axis)
    assert diag["load_converged"]
    assert state.reaction == pytest.approx(ref.load, rel=1e-8)
    assert state.contact_radius_linear == pytest.approx(ref.contact_radius, rel=0.03)
    assert state.prescribed_indentation == pytest.approx(ref.indentation, rel=0.03)
    # the constraint is penalty free: penetration sits at the ALM stopping
    # tolerance (1e-10 relative), not at a penalty compliance -- the shipped
    # penalty model is at 2.9e-2 on the same metric
    assert diag["penetration_over_indentation"] < 1e-5


def test_contact_is_noninvasive_outside_the_patch():
    ref, r_axis, z_axis = case_setup(h_over_a=0.03, depth_factor=48.0)
    ref, state, _ = solve(ref, r_axis, z_axis)
    outside = state.surface_radii > 2.0 * ref.contact_radius
    # contact membership is selected by a relative multiplier threshold, so the
    # pressure outside the patch is zero to that tolerance (not bitwise zero)
    assert np.max(state.surface_pressure[outside]) <= 1e-6 * np.max(state.surface_pressure)
    assert np.all(state.surface_gap[outside] > 0.0)
    assert np.all(state.surface_gap[state.active_nodes] > -1e-5 * ref.indentation)
    assert state.maximum_penetration < 1e-5 * ref.indentation


def test_alm_parameter_does_not_change_the_answer():
    ref, r_axis, z_axis = case_setup(h_over_a=0.03, depth_factor=48.0)
    states = [solve(ref, r_axis, z_axis, rho_factor=f)[1] for f in (0.5, 2.0)]
    assert states[0].contact_radius_linear == pytest.approx(
        states[1].contact_radius_linear, rel=2e-3
    )
    assert states[0].prescribed_indentation == pytest.approx(
        states[1].prescribed_indentation, rel=2e-3
    )


def test_load_monotonicity_and_hertz_scaling():
    """delta ~ P^(2/3): two loads must follow the Hertz exponent."""
    out = []
    for scale in (1.0, 4.0):
        ref0 = hertz_load_reference(1831.5, R, E, NU)
        ref = hertz_load_reference(ref0.load * scale, R, E, NU)
        _, r_axis, z_axis = case_setup(h_over_a=0.03, depth_factor=48.0)
        _, state, _ = solve(ref, r_axis, z_axis)
        out.append((ref, state))
    (ref1, s1), (ref2, s2) = out
    exponent = (math.log(s2.prescribed_indentation / s1.prescribed_indentation)
                / math.log(ref2.load / ref1.load))
    assert exponent == pytest.approx(2.0 / 3.0, abs=0.02)
