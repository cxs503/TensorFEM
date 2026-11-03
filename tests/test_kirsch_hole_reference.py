"""Verification of the production plane-stress path against the Kirsch solution.

The Kirsch circular-hole solution is the exact 2-D elasticity solution for an
infinite plate under uniaxial tension with a traction-free circular hole.  It is
used here as an *independent* reference: every boundary condition of the quarter
annulus used below (symmetry lines, traction-free hole, outer arc tractions) is
satisfied by the closed form itself, so the finite-element model must converge to
it and nothing is judged FE-vs-FE.

Anchors (see ``docs/kirsch-hole-stress-concentration.md``):

* the closed-form stress field satisfies radial equilibrium and the traction-free
  hole condition (self-validating oracle),
* the closed-form displacement field differentiates to the closed-form strains in
  plane stress and in plane strain (so the amplitude constants cannot drift),
* the load generator reproduces the exact arc resultant,
* the production Q4 solve reproduces the exact displacement field, the exact
  stress field at fixed physical points, and ``K_t = 3``,
* the audit's own pointwise recovery reproduces the stored production stress.

The last anchor is what makes the pointwise stress comparison meaningful: a
recovery path that does not reproduce the stored element-centre stress cannot be
used to grade the production field.
"""
import math

import numpy as np
import pytest

from tensorfem.kirsch_hole import (
    DEFAULT_SAMPLE_POINTS,
    KirschHole,
    annulus_model,
    arc_resultant,
    convergence_study,
    element_centre_cross_check,
    q4_stress_at_point,
    quarter_annulus_mesh,
    solve_case,
    symmetry_dofs,
)

HOLE = KirschHole(radius=1.0, far_field=1.0, young=1.0, poisson=0.3, plane="stress")
OUTER = 10.0
LEVELS = (16, 24, 32)


# ---------------------------------------------------------------------------
# Oracles on the closed form itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("r", [1.0, 1.1, 1.5, 2.0, 5.0, 10.0])
@pytest.mark.parametrize("theta", [0.0, 0.3, math.pi / 4, 1.2, math.pi / 2])
def test_closed_form_satisfies_radial_equilibrium(r, theta):
    """d(sigma_rr)/dr + (1/r) d(sigma_rt)/dtheta + (sigma_rr - sigma_tt)/r = 0."""
    step = 1e-6
    d_rr = (HOLE.stress_polar(r + step, theta)[0] - HOLE.stress_polar(r - step, theta)[0]) / (2 * step)
    d_rt = (HOLE.stress_polar(r, theta + step)[2] - HOLE.stress_polar(r, theta - step)[2]) / (2 * step)
    srr, stt, _ = HOLE.stress_polar(r, theta)
    residual = d_rr + d_rt / r + (srr - stt) / r
    assert abs(residual) < 1e-7


@pytest.mark.parametrize("theta", [0.0, 0.4, math.pi / 4, 1.1, math.pi / 2])
def test_hole_edge_is_traction_free(theta):
    srr, _, srt = HOLE.stress_polar(HOLE.radius, theta)
    assert abs(srr) < 1e-12
    assert abs(srt) < 1e-12


def test_stress_concentration_identities():
    """sigma_tt(a, pi/2) = 3 T and sigma_tt(a, 0) = -T exactly."""
    _, hoop_90, _ = HOLE.stress_polar(HOLE.radius, math.pi / 2)
    _, hoop_0, _ = HOLE.stress_polar(HOLE.radius, 0.0)
    assert hoop_90 == pytest.approx(3.0 * HOLE.far_field, rel=1e-15)
    assert hoop_0 == pytest.approx(-HOLE.far_field, rel=1e-15)
    assert HOLE.stress_concentration == pytest.approx(3.0)


@pytest.mark.parametrize("plane", ["stress", "strain"])
@pytest.mark.parametrize("point", [(1.3, 0.4), (2.0, 1.1), (4.0, 0.9)])
def test_displacement_matches_stress_derived_strains(plane, point):
    """Numerical differentiation of the closed-form displacement reproduces the strains."""
    hole = KirschHole(plane=plane)
    e, nu = hole.effective_constants
    r, theta = point
    step = 1e-6

    def polar(x, t):
        return hole.displacement_polar(x, t)

    ur_r = (polar(r + step, theta)[0] - polar(r - step, theta)[0]) / (2 * step)
    ur_t = (polar(r, theta + step)[0] - polar(r, theta - step)[0]) / (2 * step)
    ut_r = (polar(r + step, theta)[1] - polar(r - step, theta)[1]) / (2 * step)
    ut_t = (polar(r, theta + step)[1] - polar(r, theta - step)[1]) / (2 * step)
    ur, ut = polar(r, theta)
    strains = (ur_r, ur / r + ut_t / r, 0.5 * (ur_t / r + ut_r - ut / r))
    srr, stt, srt = hole.stress_polar(r, theta)
    expected = ((srr - nu * stt) / e, (stt - nu * srr) / e, (1.0 + nu) / e * srt)
    for got, want in zip(strains, expected):
        assert abs(got - want) < 1e-7


def test_plane_strain_matches_three_dimensional_compliance():
    """Plane strain: eps_rr = (sigma_rr - nu (sigma_tt + sigma_zz)) / E."""
    hole = KirschHole(plane="strain")
    e, nu = hole.effective_constants
    r, theta = 1.7, 0.6
    srr, stt, _ = hole.stress_polar(r, theta)
    szz = hole.poisson * (srr + stt)  # eps_zz = 0
    step = 1e-6
    ur_r = (hole.displacement_polar(r + step, theta)[0] - hole.displacement_polar(r - step, theta)[0]) / (2 * step)
    expected = (srr - hole.poisson * (stt + szz)) / hole.young
    assert ur_r == pytest.approx(expected, rel=1e-6, abs=1e-6)
    # the effective-constant route must agree with the plane-stress formula
    assert (srr - nu * stt) / e == pytest.approx(expected, rel=1e-8, abs=1e-9)


def test_mesh_is_conforming_and_counter_clockwise():
    nodes, elements, radii, thetas = quarter_annulus_mesh(HOLE, OUTER, 8, 6)
    assert nodes.shape == ((8 + 1) * (6 + 1), 2)
    assert elements.shape == (8 * 6, 4)
    coordinates = np.asarray(nodes)
    for connectivity in np.asarray(elements):
        quad = coordinates[connectivity]
        area2 = sum(
            quad[i, 0] * quad[(i + 1) % 4, 1] - quad[(i + 1) % 4, 0] * quad[i, 1] for i in range(4)
        )
        assert area2 > 0.0, "element orientation must be counter-clockwise"
    assert radii[0] == pytest.approx(HOLE.radius)
    assert radii[-1] == pytest.approx(OUTER)
    assert thetas[0] == 0.0 and thetas[-1] == pytest.approx(math.pi / 2)


def test_consistent_loads_match_exact_arc_resultant():
    for n_theta in (8, 16, 32):
        model, meta = annulus_model(HOLE, OUTER, n_theta, n_theta)
        applied = meta["forces"].numpy()
        exact = arc_resultant(HOLE, OUTER)
        assert applied[0::2].sum() == pytest.approx(exact[0], rel=2e-3)
        assert abs(applied[1::2].sum()) < 5e-2 * abs(exact[0])


def test_symmetry_dofs_cover_both_cut_lines():
    fixed = symmetry_dofs(4, 3)
    assert len(fixed) == 2 * (3 + 1)  # no node is shared by the two cut lines
    assert all(dof < 2 * (4 + 1) * (3 + 1) for dof in fixed)
    # u_y (odd dofs) is fixed on theta = 0, u_x (even dofs) on theta = pi/2
    assert sum(1 for dof in fixed if dof % 2 == 1) == 3 + 1
    assert sum(1 for dof in fixed if dof % 2 == 0) == 3 + 1


# ---------------------------------------------------------------------------
# Production solve against the closed form
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def study():
    return convergence_study(levels=LEVELS, outer_radius=OUTER)


def test_displacement_converges_to_closed_form(study):
    errors = [case["displacement_error"]["l2_relative"] for case in study["cases"]]
    assert errors[-1] < 5e-3
    assert all(fine < coarse for coarse, fine in zip(errors, errors[1:]))
    orders = [order["displacement_order"] for order in study["observed_orders"]]
    assert all(1.5 < order < 2.5 for order in orders), orders


@pytest.mark.parametrize("index", range(len(LEVELS)))
def test_pointwise_stress_within_three_percent(study, index):
    case = study["cases"][index]
    for row in case["stress_rows"]:
        assert row["inside"], row
        assert row["hoop_relative_error"] < 0.03, row


def test_stress_concentration_factor_recovered(study):
    finest = study["cases"][-1]["hole_edge"]
    assert finest["hole_edge_at_90_degrees"] == pytest.approx(3.0, rel=0.01)
    assert finest["hole_edge_at_0_degrees"] == pytest.approx(-1.0, rel=0.05)
    assert finest["peak_theta_degrees"] == pytest.approx(90.0, abs=30.0)


def test_recovery_path_reproduces_stored_stress(study):
    for case in study["cases"]:
        check = case["recovery_cross_check"]
        assert check["passed"], check
        assert check["max_relative_difference"] < 1e-10


def test_load_generator_and_equilibrium(study):
    for case in study["cases"]:
        balance = case["load_resultant"]
        assert balance["relative_error"] < 5e-3
        assert balance["reaction"][0] == pytest.approx(-balance["applied"][0], rel=1e-9)
        assert balance["reaction"][1] == pytest.approx(-balance["applied"][1], abs=1e-9)


def test_stress_error_improves_with_refinement(study):
    errors = [case["max_stress_relative_error"] for case in study["cases"]]
    assert errors[-1] < errors[0]
    assert errors[-1] < 0.01


def test_solution_is_linear_in_the_far_field():
    """Doubling the tension must double both the displacement and the stress field."""
    base = solve_case(HOLE, OUTER, n_theta=16)
    scaled = solve_case(KirschHole(far_field=2.0), OUTER, n_theta=16)
    assert np.allclose(scaled["displacement"], 2.0 * base["displacement"], rtol=1e-12, atol=1e-12)
    for row, doubled in zip(base["stress_rows"], scaled["stress_rows"]):
        assert doubled["hoop_fe"] == pytest.approx(2.0 * row["hoop_fe"], rel=1e-12)


def test_zero_load_gives_zero_field():
    case = solve_case(KirschHole(far_field=0.0), OUTER, n_theta=12)
    assert np.max(np.abs(case["displacement"])) < 1e-14
    assert case["hole_edge"]["peak_hoop"] == pytest.approx(0.0, abs=1e-12)


def test_point_sampling_rejects_singular_radius():
    with pytest.raises(ValueError):
        HOLE.stress_polar(0.0, 0.0)


def test_parameters_are_validated():
    with pytest.raises(ValueError):
        KirschHole(radius=-1.0)
    with pytest.raises(ValueError):
        KirschHole(plane="axisymmetric")
    with pytest.raises(ValueError):
        quarter_annulus_mesh(HOLE, OUTER, 0, 4)
    with pytest.raises(ValueError):
        quarter_annulus_mesh(HOLE, 0.5, 4, 4)


def test_sample_points_are_declared():
    """The published comparison set is fixed; changing it must be a deliberate edit."""
    assert DEFAULT_SAMPLE_POINTS[0][0] == pytest.approx(1.02)
    assert DEFAULT_SAMPLE_POINTS[-1][0] == pytest.approx(2.0)


def test_element_centre_cross_check_detects_bad_recovery():
    """Negative control: perturbing the stored stress must break the cross-check."""
    case = solve_case(HOLE, OUTER, n_theta=12)
    stored = case["stored_stress"].copy()
    stored[0] *= 1.05
    result = element_centre_cross_check(case["meta"], case["displacement"], stored)
    assert not result["passed"]
    assert result["max_relative_difference"] > 1e-3


def test_pointwise_recovery_matches_centre_value_at_centre():
    case = solve_case(HOLE, OUTER, n_theta=12)
    meta = case["meta"]
    nodes = np.asarray(meta["nodes"])
    connectivity = np.asarray(meta["elements"])[7]
    centre = nodes[connectivity].mean(axis=0)
    sampled = q4_stress_at_point(meta, case["displacement"], centre)
    assert sampled["inside"]
    assert np.allclose(sampled["stress"], case["stored_stress"][7], rtol=1e-10, atol=1e-12)