"""Verification of the Boussinesq far-field kernel and its use as absorbing data.

Three independent layers:

1. the point-load kernel against the textbook surface values (Boussinesq 1885;
   Poulos & Davis 1974 §3.2);
2. the axisymmetric superposition against closed forms for a uniformly loaded
   disc -- centre, edge and the on-axis-at-depth integral;
3. the horizontal (radial) component against a brute-force Cartesian summation
   over the disc, which shares no algebra with the azimuthal projection and so
   catches a wrong ``(r - a cos theta)/s`` factor that the axial closed forms
   cannot see.

The solver-level part is marked ``slow`` (two load-controlled solves).
"""

import math

import numpy as np
import pytest

from tensorfem.boussinesq_farfield import (
    disc_axis_settlement,
    disc_center_settlement,
    disc_edge_settlement,
    effective_compliance,
    nodal_force_displacement,
    point_load_displacement,
    ring_load_displacement,
    uniform_disc_displacement,
)
from tensorfem.hertz_halfspace_loadcontrol import (
    boussinesq_far_field_dofs,
    graded_depth,
    hertz_load_reference,
    solve_axisymmetric_contact_load_control,
    two_zone_axis,
)

E, NU = 1.0, 0.3


def cartesian_disc_displacement(radius, pressure, field_r, field_depth, samples=600):
    """Brute-force point-load superposition over a disc in Cartesian coordinates."""
    coords = np.linspace(-radius, radius, samples)
    x, y = np.meshgrid(coords, coords)
    inside = (x**2 + y**2) <= radius**2
    x, y = x[inside], y[inside]
    cell = (coords[1] - coords[0]) ** 2
    u_r = 0.0
    u_z = 0.0
    for load_x, load_y in zip(x, y):
        dx = field_r - load_x
        dy = 0.0 - load_y
        s = math.hypot(dx, dy)
        if s == 0.0:
            continue
        # point-load kernel evaluated at horizontal distance s, depth d
        u_s, u_z_pt = point_load_displacement(
            np.array([s]), np.array([field_depth]), load=1.0, young=E, poisson=NU
        )
        # horizontal displacement points from the load to the field point
        u_r += u_s[0] * (dx / s) * pressure * cell
        u_z += u_z_pt[0] * pressure * cell
    return u_r, u_z


def test_effective_compliance_matches_shear_modulus():
    shear = E / (2 * (1 + NU))
    assert effective_compliance(E, NU) == pytest.approx(1.0 / (4 * math.pi * shear))


def test_point_load_surface_closed_forms():
    r = np.array([0.5, 1.0, 3.0])
    u_r, u_z = point_load_displacement(r, np.zeros(3), load=1.0, young=E, poisson=NU)
    assert u_z == pytest.approx((1 - NU**2) / (math.pi * E * r), rel=1e-12)
    assert u_r == pytest.approx(
        -(1 - 2 * NU) * (1 + NU) / (2 * math.pi * E * r), rel=1e-12
    )


def test_point_load_on_axis_at_depth_closed_form():
    depth = np.array([0.25, 1.0, 4.0])
    _, u_z = point_load_displacement(np.zeros(3), depth, load=1.0, young=E, poisson=NU)
    expected = (1 + NU) * (3 - 2 * NU) / (2 * math.pi * E * depth)
    assert u_z == pytest.approx(expected, rel=1e-12)


def test_point_load_rejects_negative_depth():
    with pytest.raises(ValueError):
        point_load_displacement(np.array([1.0]), np.array([-1.0]))


def test_ring_matches_point_load_in_the_limit_of_zero_radius():
    ur_ring, uz_ring = ring_load_displacement(
        np.array([2.0]), np.array([0.5]), 1e-7, 1.0, E, NU, n_theta=8
    )
    ur_pt, uz_pt = point_load_displacement(
        np.array([2.0]), np.array([0.5]), load=1.0, young=E, poisson=NU
    )
    assert ur_ring[0] == pytest.approx(ur_pt[0], rel=1e-6)
    assert uz_ring[0] == pytest.approx(uz_pt[0], rel=1e-6)


def test_uniform_disc_centre_and_edge_settlement():
    a, p = 0.5, 3.0
    _, u_z = uniform_disc_displacement(
        np.array([0.0, a]), np.array([0.0, 0.0]), a, p, E, NU, n_r=200, n_theta=32
    )
    assert u_z[0] == pytest.approx(disc_center_settlement(a, p, E, NU), rel=1e-10)
    # the edge integral is only log-singular; the quadrature converges slowly
    assert u_z[1] == pytest.approx(disc_edge_settlement(a, p, E, NU), rel=2e-3)


def test_uniform_disc_on_axis_at_depth_closed_form():
    a, p = 0.5, 3.0
    depths = np.array([0.0, 0.1, 1.0, 5.0])
    _, u_z = uniform_disc_displacement(
        np.zeros(depths.size), depths, a, p, E, NU, n_r=256, n_theta=32
    )
    expected = np.array([disc_axis_settlement(a, p, d, E, NU) for d in depths])
    assert u_z == pytest.approx(expected, rel=1e-10)


def test_disc_axis_settlement_reduces_to_centre_value():
    a, p = 0.4, 2.0
    assert disc_axis_settlement(a, p, 0.0, E, NU) == pytest.approx(
        disc_center_settlement(a, p, E, NU), rel=1e-12
    )


@pytest.mark.parametrize("field_r, field_depth", [(0.9, 0.0), (2.0, 0.3), (0.0, 0.7)])
def test_radial_component_matches_brute_force_cartesian_sum(field_r, field_depth):
    """The azimuthal projection ``(r - a cos theta)`` cross-checked independently."""
    a, p = 0.5, 1.0
    u_r, u_z = uniform_disc_displacement(
        np.array([field_r]), np.array([field_depth]), a, p, E, NU, n_r=96, n_theta=48
    )
    ref_r, ref_z = cartesian_disc_displacement(a, p, field_r, field_depth, samples=420)
    assert u_r[0] == pytest.approx(ref_r, rel=6e-3, abs=1e-9)
    assert u_z[0] == pytest.approx(ref_z, rel=6e-3)


def test_nodal_force_superposition_reproduces_the_disc():
    a, p = 0.5, 2.0
    # midpoint ring discretisation: sum(ring_force) == p pi a^2 exactly, so the
    # only difference from the Gauss disc is the radial resolution
    edges = np.linspace(0.0, a, 65)
    ring_radii = 0.5 * (edges[:-1] + edges[1:])
    ring_force = p * 2 * np.pi * ring_radii * (edges[1] - edges[0])
    field_r, field_depth = 2.0, 0.4
    u_r, u_z = nodal_force_displacement(
        np.array([field_r]), np.array([field_depth]), ring_radii, ring_force, E, NU
    )
    ref_r, ref_z = uniform_disc_displacement(
        np.array([field_r]), np.array([field_depth]), a, p, E, NU, n_r=256, n_theta=48
    )
    assert u_z[0] == pytest.approx(ref_z[0], rel=2e-3)
    assert u_r[0] == pytest.approx(ref_r[0], rel=2e-3)


def test_far_field_prescription_is_zero_for_a_contact_free_state():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, -1.0], [1.0, -1.0]])
    radii = np.array([0.0, 1.0])
    areas = np.array([math.pi / 2, 3 * math.pi / 2])
    prescribed = boussinesq_far_field_dofs(
        ndof=8,
        nodes=nodes,
        bottom=np.array([2, 3]),
        outer=np.array([1, 3]),
        radii=radii,
        areas=areas,
        pressure=np.zeros(2),
        young=E,
        poisson=NU,
    )
    assert prescribed.shape == (8,)
    assert np.all(prescribed == 0.0)


def test_far_field_prescription_points_down_and_inwards():
    """A compressive contact patch pushes the far boundary down, and the surface
    far field is drawn towards the load (Boussinesq ``u_r < 0`` for ``r > 0``)."""
    nodes = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, -6.0], [2.0, -6.0]])
    radii = np.array([0.0, 2.0])
    areas = np.array([math.pi, 3 * math.pi])
    pressure = np.array([1.0, 0.0])
    prescribed = boussinesq_far_field_dofs(
        ndof=8,
        nodes=nodes,
        bottom=np.array([2, 3]),
        outer=np.array([1, 3]),
        radii=radii,
        areas=areas,
        pressure=pressure,
        young=E,
        poisson=NU,
        outer_vertical=True,
    ).reshape(4, 2)
    # the axis node is a symmetry/surface node: not a far-field node
    assert prescribed[0, 0] == 0.0 and prescribed[0, 1] == 0.0
    # outer surface node: pulled inwards and pushed down
    assert prescribed[1, 0] < 0.0
    assert prescribed[1, 1] < 0.0
    # bottom boundary: pushed down everywhere, and the load at the axis makes
    # the axis-bottom point move radially outwards (u_r = 0 only at r = 0 by
    # symmetry, so the on-axis kernel gives exactly zero there)
    assert prescribed[2, 1] < 0.0
    assert prescribed[2, 0] == 0.0
    assert prescribed[3, 1] > prescribed[2, 1]  # |u_z| decays off the axis


R, YOUNG = 10.0, 1.0e5


def _case(a_over_r=0.05, h_over_a=0.03, depth_factor=24.0):
    ref = hertz_load_reference(
        4.0 * (YOUNG / (1 - NU**2)) * (a_over_r * R) ** 3 / (3.0 * R), R, YOUNG, NU
    )
    a = ref.contact_radius
    r_axis = two_zone_axis(3 * a, depth_factor * a, int(round(3 / h_over_a)), 20)
    z_axis = graded_depth(depth_factor * a, 4 * a, 18, 12)
    return ref, r_axis, z_axis


@pytest.mark.slow
def test_absorbing_far_field_removes_the_truncation_bias_on_delta():
    """Same mesh, same load: the clamped patch is stiff, the half-space is not."""
    ref, r_axis, z_axis = _case()
    results = {}
    for mode in ("zero", "boussinesq"):
        _, state, diag = solve_axisymmetric_contact_load_control(
            load=ref.load,
            radius=R,
            young=YOUNG,
            poisson=NU,
            r_axis=r_axis,
            z_axis=z_axis,
            load_tol=1e-10,
            far_field=mode,
        )
        results[mode] = (state, diag)
    clamped = 100.0 * (results["zero"][0].prescribed_indentation / ref.indentation - 1.0)
    absorbing = 100.0 * (
        results["boussinesq"][0].prescribed_indentation / ref.indentation - 1.0
    )
    # the clamped patch underestimates delta (it holds the far field in place)
    assert clamped < -1.0
    assert abs(absorbing) < abs(clamped)
    assert abs(absorbing) < 1.0
    # the contact radius is a near-field quantity and must not move
    a_clamped = results["zero"][0].contact_radius_linear
    a_absorbing = results["boussinesq"][0].contact_radius_linear
    assert abs(a_absorbing / ref.contact_radius - a_clamped / ref.contact_radius) * 100 < 0.05
    # the boundary/traction fixed point is reached and declared
    ff = results["boussinesq"][1]["far_field"]
    assert ff["mode"] == "boussinesq"
    assert ff["converged"]
    assert ff["history"][-1]["relative_update"] <= ff["tolerance"]
    # constraint quality is untouched by the boundary treatment
    assert results["boussinesq"][1]["penetration_over_indentation"] < 1e-5
    assert results["boussinesq"][1]["load_residual"] < 1e-9


def test_zero_far_field_is_the_historical_model():
    """The default path must stay bit-for-bit the clamped model."""
    ref, r_axis, z_axis = _case(a_over_r=0.05, h_over_a=0.03, depth_factor=12.0)
    kw = dict(load=ref.load, radius=R, young=YOUNG, poisson=NU, r_axis=r_axis, z_axis=z_axis)
    _, s_default, d_default = solve_axisymmetric_contact_load_control(**kw)
    _, s_zero, d_zero = solve_axisymmetric_contact_load_control(far_field="zero", **kw)
    assert s_default.prescribed_indentation == s_zero.prescribed_indentation
    assert d_default["far_field"]["mode"] == "zero"
    assert d_default["far_field"]["passes"] == 0