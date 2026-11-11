"""Verification of the 8-node three-dimensional cohesive interface element.

The checks are ordered as a ladder, from constitutive exactness up to an
assembled three-dimensional equilibrium check. Every quantitative assertion
compares against a closed form, an independent quadrature, or an independent
element (the two-node mode-I law of :mod:`tensorfem.cohesive`).
"""

import math

import pytest
import torch

from tensorfem.cohesive import BilinearCohesiveLaw
from tensorfem.cohesive3d import (
    Cohesive3DLaw,
    cohesive_interface_response,
    face_shape,
    gauss_rule,
    interface_area,
    interface_frame,
)

D = torch.float64

KN, KS = 1.0e5, 6.0e4
TN0, TS0 = 10.0, 8.0
GIC, GIIC = 0.5, 1.2
ETA = 2.284


def law() -> Cohesive3DLaw:
    return Cohesive3DLaw(KN, KS, TN0, TS0, GIC, GIIC, ETA)


def flat_square(side: float = 1.0) -> torch.Tensor:
    """Unit (or scaled) flat interface in the z=0 / z=h planes."""
    quad = torch.tensor(
        [[0.0, 0.0], [side, 0.0], [side, side], [0.0, side]], dtype=D
    )
    bottom = torch.cat([quad, torch.zeros(4, 1, dtype=D)], dim=1)
    top = torch.cat([quad, torch.full((4, 1), 1.0, dtype=D)], dim=1)
    return torch.stack((bottom, top))


def distorted_interface() -> torch.Tensor:
    """Warped, non-planar quadrilateral interface with unequal face shapes."""
    quad = torch.tensor(
        [[0.0, 0.0], [1.3, 0.05], [1.05, 0.9], [-0.1, 1.15]], dtype=D
    )
    bottom = torch.cat([quad, torch.zeros(4, 1, dtype=D)], dim=1)
    top = bottom.clone()
    top[:, 2] = torch.tensor([0.9, 1.02, 1.1, 0.85], dtype=D)
    top[:, 0] += torch.tensor([0.02, -0.01, 0.015, -0.02], dtype=D)
    return torch.stack((bottom, top))


def uniform(opening: torch.Tensor) -> torch.Tensor:
    """Nodal displacements producing a uniform local separation vector."""
    u = torch.zeros(8, 3, dtype=D)
    u[4:, 0] = opening[1]
    u[4:, 1] = opening[2]
    u[4:, 2] = opening[0]
    return u


def internal_work_along_direction(law_, direction, amplitude, steps=4000):
    """Numeric integral of t . d(delta) along a proportional path (unit area)."""
    c = torch.as_tensor(direction, dtype=D)
    c = c / torch.linalg.vector_norm(c)
    grid = torch.linspace(0.0, amplitude, steps, dtype=D)
    values = torch.zeros_like(grid)
    history = torch.zeros((), dtype=D)
    for i, a in enumerate(grid):
        separation = a * c
        values[i] = law_.traction(separation, history) @ c
        history = torch.maximum(history, a)
    return torch.trapezoid(values, grid)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def test_frame_is_orthonormal_and_area_is_exact():
    coords = distorted_interface()
    for xi, eta in ((0.0, 0.0), (-0.7, 0.4), (0.9, -0.9)):
        frame, area = interface_frame(coords, xi, eta)
        assert float(area) > 0.0
        assert torch.allclose(frame @ frame.T, torch.eye(3, dtype=D), atol=1e-12)

    # The bilinear, non-planar face area is integrated exactly by order-3 rules
    # only to quadrature accuracy; compare two nested rules for convergence.
    coarse = interface_area(flat_square(), order=2)
    fine = interface_area(flat_square(), order=3)
    assert abs(float(coarse) - 1.0) < 1e-14
    assert abs(float(fine) - 1.0) < 1e-14


def test_degenerate_interface_raises():
    coords = flat_square()
    collapsed = coords.clone()
    collapsed[:, :, 0] = 0.0  # mid-surface collapses to a line: zero Jacobian
    with pytest.raises(ValueError):
        interface_frame(collapsed, 0.0, 0.0)


def test_input_validation():
    law_ = law()
    with pytest.raises(ValueError):
        cohesive_interface_response(torch.zeros(5, 3, dtype=D), torch.zeros(24, dtype=D), law_)
    with pytest.raises(ValueError):
        cohesive_interface_response(flat_square(), torch.zeros(24, dtype=D), law_, history=torch.zeros(3, dtype=D))
    with pytest.raises(ValueError):
        Cohesive3DLaw(KN, KS, TN0, TS0, -1.0, GIIC, ETA)
    with pytest.raises(ValueError):
        law_.direction_measures(torch.zeros(3, dtype=D))


# ---------------------------------------------------------------------------
# constitutive ladder: exact closed forms
# ---------------------------------------------------------------------------
def test_pure_mode_i_matches_two_node_law_and_closed_form():
    """Uniform mode-I opening reproduces the 2-D mode-I law exactly."""
    law_ = law()
    reference = BilinearCohesiveLaw(KN, TN0, GIC)
    coords = flat_square()
    worst = 0.0
    for opening in (0.5e-4, 1.0e-4, 1.5e-4, 5e-4, 0.0):
        result = cohesive_interface_response(coords, uniform(torch.tensor([opening, 0.0, 0.0], dtype=D)), law_)
        expected = reference.envelope_traction(torch.tensor(opening, dtype=D))
        nodal_force = result.force.reshape(8, 3)[4:, 2]
        analytic_force = expected * float(result.area) / 4.0
        worst = max(worst, float(torch.max(torch.abs(nodal_force - analytic_force))))
        # analytic envelope, bad area scaling
        assert torch.allclose(nodal_force, analytic_force, rtol=1e-13, atol=1e-13)
    assert worst < 1e-12


def test_pure_shear_modes_hit_strength_and_failure():
    """Mode II / mode III: peak traction ts0 at delta_s = ts0/ks, zero at 2 GIIc/ts0."""
    law_ = law()
    coords = flat_square()
    onset = TS0 / KS
    failure = 2.0 * GIIC / TS0
    for axis in (1, 2):  # in-plane tangent 1 (x) and tangent 2 (y)
        separation = torch.zeros(3, dtype=D)
        separation[axis] = onset
        peak = cohesive_interface_response(coords, uniform(separation), law_)
        assert abs(float(peak.traction[0, axis]) - TS0) < 1e-10

        separation[axis] = failure
        failed = cohesive_interface_response(coords, uniform(separation), law_)
        assert float(torch.max(torch.abs(failed.traction))) < 1e-12
        assert float(torch.max(failed.damage)) > 1.0 - 1e-12


def test_bk_critical_energy_matches_published_criterion():
    law_ = law()
    for direction, shear_fraction in (
        (torch.tensor([1.0, 0.0, 0.0], dtype=D), 0.0),
        (torch.tensor([0.0, 1.0, 0.0], dtype=D), 1.0),
        (torch.tensor([0.0, 0.0, 1.0], dtype=D), 1.0),
        (torch.tensor([1.0, 1.0, 0.0], dtype=D), KS / (KN + KS)),
        (torch.tensor([1.0, 2.0, 0.0], dtype=D), 4.0 * KS / (KN + 4.0 * KS)),
    ):
        _, _, gc, _ = law_.direction_measures(direction)
        expected = GIC + (GIIC - GIC) * shear_fraction**ETA
        assert abs(float(gc) - expected) < 1e-13 * expected + 1e-14

    # unit directions with mixed normal/shear and both shear components
    direction = torch.tensor([1.0, 1.0, 1.0], dtype=D)
    c = direction / torch.linalg.vector_norm(direction)
    work_stiffness = KN * c[0] ** 2 + KS * (c[1] ** 2 + c[2] ** 2)
    shear_fraction = KS * (c[1] ** 2 + c[2] ** 2) / work_stiffness
    expected = GIC + (GIIC - GIC) * float(shear_fraction) ** ETA
    _, _, gc, _ = law_.direction_measures(direction)
    assert abs(float(gc) - expected) < 1e-13


def test_mixed_mode_work_integral_equals_critical_energy():
    """int t . d(delta) along a proportional path equals Gc(direction)."""
    law_ = law()
    for direction in (
        torch.tensor([1.0, 0.0, 0.0], dtype=D),
        torch.tensor([0.0, 3.0, 4.0], dtype=D),
        torch.tensor([1.0, 1.0, 1.0], dtype=D),
        torch.tensor([4.0, 3.0, 0.0], dtype=D),
    ):
        lam0, lamf, gc, work_stiffness = law_.direction_measures(direction)
        # analytic envelope area w * lam0 * lamf / 2 == Gc by construction
        assert abs(float(0.5 * work_stiffness * lam0 * lamf) - float(gc)) < 1e-13 * float(gc)
        work = internal_work_along_direction(law_, direction, float(lamf) * 1.05)
        # trapezoidal integration of the piecewise-linear envelope, quadrature limited
        assert abs(float(work) - float(gc)) / float(gc) < 1e-4


def test_element_dissipated_energy_equals_gc_times_area():
    law_ = law()
    direction = torch.tensor([1.0, 1.0, 0.0], dtype=D)
    _, lamf, gc, _ = law_.direction_measures(direction)
    unit = direction / torch.linalg.vector_norm(direction)
    coords = flat_square(side=2.0)
    result = cohesive_interface_response(coords, uniform(unit * (float(lamf) * 1.01)), law_)
    expected = float(gc) * float(result.area)
    assert abs(float(result.dissipated_energy) - expected) / expected < 1e-12


# ---------------------------------------------------------------------------
# element mechanics
# ---------------------------------------------------------------------------
def test_consistent_tangent_matches_central_differences():
    """Element tangent equals dF/du from central differences, on both branches."""
    law_ = law()
    coords = distorted_interface()
    generator = torch.Generator().manual_seed(20261111)
    for scale in (2e-5, 3e-4):  # elastic branch, then softening branch
        u = scale * torch.rand(8, 3, dtype=D, generator=generator)
        u[4:, 2] += scale * 2.0
        result = cohesive_interface_response(coords, u, law_)
        numeric = torch.zeros((24, 24), dtype=D)
        step = 1e-9 * max(scale, 1.0)
        flat = u.reshape(24)
        for k in range(24):
            plus = flat.clone()
            minus = flat.clone()
            plus[k] += step
            minus[k] -= step
            f_plus = cohesive_interface_response(coords, plus, law_).force
            f_minus = cohesive_interface_response(coords, minus, law_).force
            numeric[:, k] = (f_plus - f_minus) / (2.0 * step)
        scale_factor = float(torch.max(torch.abs(result.stiffness)))
        error = float(torch.max(torch.abs(result.stiffness - numeric))) / scale_factor
        assert error < 1e-6, f"tangent mismatch {error:.3e}"


def test_rigid_translation_is_stress_free():
    law_ = law()
    coords = distorted_interface()
    for translation in (
        torch.tensor([1e-3, 0.0, 0.0], dtype=D),
        torch.tensor([0.0, 0.0, 5e-4], dtype=D),
        torch.tensor([2e-4, -3e-4, 1e-4], dtype=D),
    ):
        u = torch.zeros(8, 3, dtype=D)
        u += translation
        result = cohesive_interface_response(coords, u, law_)
        assert float(torch.max(torch.abs(result.force))) < 1e-12
        assert float(torch.max(torch.abs(result.traction))) < 1e-12


def test_action_reaction_and_moment_equilibrium():
    law_ = law()
    coords = distorted_interface()
    u = 1.0e-5 * torch.tensor(
        [[0.1, 0.0, 0.3], [0.0, 0.2, 0.4], [-0.3, 0.1, 0.2], [0.2, -0.2, 0.5],
         [0.4, 0.1, 0.8], [0.2, 0.3, 0.9], [-0.1, 0.2, 0.7], [0.5, -0.1, 1.0]],
        dtype=D,
    )
    result = cohesive_interface_response(coords, u, law_)
    nodal = result.force.reshape(8, 3)
    assert torch.allclose(nodal.sum(dim=0), torch.zeros(3, dtype=D), atol=1e-10)
    # upper-face resultant equals an independently integrated traction field
    points, weights = gauss_rule(3)
    expected = torch.zeros(3, dtype=D)
    for xi, wx in zip(points, weights):
        for eta, wy in zip(points, weights):
            shape, _ = face_shape(xi, eta)
            frame, area = interface_frame(coords, xi, eta)
            separation_global = shape @ (u[4:] - u[:4])
            traction = law_.traction(frame @ separation_global, torch.zeros((), dtype=D))
            expected += wx * wy * area * (frame.T @ traction)
    # Independent re-implementation with the same Gauss order: the element
    # assembly (interpolation, frame rotation, signs) must match to round-off.
    same_order = torch.zeros(3, dtype=D)
    points2, weights2 = gauss_rule(2)
    for xi, wx in zip(points2, weights2):
        for eta, wy in zip(points2, weights2):
            shape, _ = face_shape(xi, eta)
            frame, area = interface_frame(coords, xi, eta)
            separation_global = shape @ (u[4:] - u[:4])
            traction = law_.traction(frame @ separation_global, torch.zeros((), dtype=D))
            same_order += wx * wy * area * (frame.T @ traction)
    assert torch.allclose(nodal[4:].sum(dim=0), same_order, rtol=1e-13, atol=1e-16)

    # A higher-order rule on a warped interface differs only by the quadrature
    # error of the (non-polynomial) mid-surface frame: order 1e-6 here.
    assert torch.allclose(nodal[4:].sum(dim=0), expected, rtol=1e-5, atol=1e-12)

    # on a flat interface the local frame is constant, so order-2 quadrature is
    # algebraically exact and the resultant must match to machine precision
    flat = flat_square()
    flat_result = cohesive_interface_response(flat, u, law_)
    flat_points, flat_weights = gauss_rule(3)
    flat_expected = torch.zeros(3, dtype=D)
    for xi, wx in zip(flat_points, flat_weights):
        for eta, wy in zip(flat_points, flat_weights):
            shape, _ = face_shape(xi, eta)
            frame, area = interface_frame(flat, xi, eta)
            separation_global = shape @ (u[4:] - u[:4])
            traction = law_.traction(frame @ separation_global, torch.zeros((), dtype=D))
            flat_expected += wx * wy * area * (frame.T @ traction)
    assert torch.allclose(
        flat_result.force.reshape(8, 3)[4:].sum(dim=0), flat_expected, rtol=1e-12, atol=1e-14
    )


def test_bilinear_separation_field_is_integrated_exactly():
    """Order-2 Gauss reproduces exactly what an order-3 rule gives."""
    law_ = law()
    coords = flat_square()
    # nodal displacements chosen so the separation field is bilinear in (xi,eta);
    # an elastic-branch field keeps the integrand polynomial of degree <= 3.
    nodal_opening = 1e-5 * torch.tensor(
        [[0.0, 0.0, 0.4], [1.0, 0.2, 0.5], [1.4, 1.1, 0.6], [-0.2, 1.0, 0.45]],
        dtype=D,
    )
    u = torch.zeros(8, 3, dtype=D)
    u[4:] = nodal_opening
    coarse = cohesive_interface_response(coords, u, law_, order=2)
    fine = cohesive_interface_response(coords, u, law_, order=3)
    assert float(torch.max(torch.abs(coarse.force - fine.force))) < 1e-14
    assert float(torch.max(torch.abs(coarse.stiffness - fine.stiffness))) < 1e-8


def test_history_is_irreversible_with_secant_unloading():
    law_ = law()
    coords = flat_square()
    peak_opening = 1.6e-4  # beyond onset (1e-4), inside the softening branch
    loaded = cohesive_interface_response(coords, uniform(torch.tensor([peak_opening, 0.0, 0.0], dtype=D)), law_)
    recorded = loaded.maximum_separation.clone()
    peak_traction = loaded.traction[0, 0]
    damage_loaded = loaded.damage[0]

    # unload to 40% of the peak opening: secant stiffness, no history loss
    unload = 0.4 * peak_opening
    unloaded = cohesive_interface_response(
        coords, uniform(torch.tensor([unload, 0.0, 0.0], dtype=D)), law_, history=recorded
    )
    secant = float(peak_traction) / peak_opening
    assert abs(float(unloaded.traction[0, 0]) - secant * unload) < 1e-12
    assert torch.allclose(unloaded.maximum_separation, recorded, atol=1e-18)

    # reloading to the same opening reproduces the envelope traction
    reloaded = cohesive_interface_response(
        coords, uniform(torch.tensor([peak_opening, 0.0, 0.0], dtype=D)), law_, history=recorded
    )
    assert abs(float(reloaded.traction[0, 0]) - float(peak_traction)) < 1e-12
    assert abs(float(reloaded.damage[0]) - float(damage_loaded)) < 1e-14

    # after full failure the interface carries no traction even on reload
    failed = cohesive_interface_response(
        coords, uniform(torch.tensor([0.2, 0.0, 0.0], dtype=D)), law_, history=recorded
    )
    assert float(torch.max(torch.abs(failed.traction))) < 1e-12


def test_normal_compression_carries_penalty_without_damage():
    law_ = law()
    coords = flat_square()
    compression = -3e-4
    result = cohesive_interface_response(coords, uniform(torch.tensor([compression, 0.0, 0.0], dtype=D)), law_)
    expected = KN * compression
    assert abs(float(result.traction[0, 0]) - expected) < 1e-9
    assert float(torch.max(torch.abs(result.damage))) < 1e-15
    assert float(torch.max(result.maximum_separation)) < 1e-12


def test_damage_is_normalised_and_reaches_one_at_failure():
    law_ = law()
    coords = flat_square()
    failure = 2.0 * GIC / TN0
    result = cohesive_interface_response(coords, uniform(torch.tensor([failure, 0.0, 0.0], dtype=D)), law_)
    assert float(torch.max(result.damage)) > 1.0 - 1e-12
    elastic = cohesive_interface_response(coords, uniform(torch.tensor([1e-6, 0.0, 0.0], dtype=D)), law_)
    assert float(torch.max(elastic.damage)) < 1e-12


def test_stiffness_symmetry_and_shape():
    law_ = law()
    coords = distorted_interface()
    result = cohesive_interface_response(coords, 1e-5 * torch.rand(8, 3, dtype=D, generator=torch.Generator().manual_seed(7)), law_)
    assert result.stiffness.shape == (24, 24)
    scale = float(torch.max(torch.abs(result.stiffness)))
    assert float(torch.max(torch.abs(result.stiffness - result.stiffness.T))) < 1e-10 * scale


def test_residual_force_equals_tangent_times_zero_after_convergence():
    """Sanity: f(u*) = 0 along a homogeneous mode-I path that is traction free."""
    law_ = law()
    coords = flat_square()
    result = cohesive_interface_response(coords, torch.zeros(8, 3, dtype=D), law_)
    assert float(torch.max(torch.abs(result.force))) < 1e-15
    assert float(torch.linalg.matrix_norm(result.stiffness)) > 0.0


def test_face_shape_partition_of_unity():
    for xi, eta in ((0.0, 0.0), (0.3, -0.8), (1.0, 1.0)):
        shape, derivatives = face_shape(xi, eta)
        assert abs(float(shape.sum()) - 1.0) < 1e-15
        assert abs(float(derivatives.sum(dim=0)[0])) < 1e-15
        assert abs(float(derivatives.sum(dim=0)[1])) < 1e-15


def test_gauss_weights_integrate_polynomials_exactly():
    for order, degree in ((1, 1), (2, 3), (3, 5)):
        points, weights = gauss_rule(order)
        for exponent in (0, degree):
            integral = sum(
                w * p**exponent for p, w in zip(points, weights)
            )
            exact = 2.0 / (exponent + 1) if exponent % 2 == 0 else 0.0
            assert math.isclose(integral, exact, rel_tol=1e-14, abs_tol=1e-14)