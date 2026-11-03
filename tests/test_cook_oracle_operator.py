"""Regression tests for the Cook-membrane audit operators (m2430).

Context: the merged audit of 2026-11-02 left two open items - a pointwise
recovery that disagreed with the stored Q4 stress by hundreds of percent, and a
loaded-edge deflection profile where the independent Q8 reference (7.6 -> 51.6)
contradicted the production Q4 field (23.2 -> 25.0).  Both trace to the audit's
own ``q4_strain_matrix`` / ``q8_strain_matrix``: the physical gradient was formed
as ``dN @ inv(J)`` instead of ``dN @ inv(J).T``, which is only equivalent when the
Jacobian is symmetric (rectangles) and is wrong on the tapered Cook map, and the
Q4 parent-derivative table additionally used ``eta`` / ``xi`` where ``1 + eta`` /
``1 + xi`` belong, leaving the measure 16x too large.

The tests below pin the operators with *convention-free* checks (constant-strain
patch, constant-strain energy identity, finite-difference gradients) so the same
defect cannot return, and pin the honest negative result that the production Q4
element-centre stress field is still not certified to 3 % against the oracle.
"""

import math

import numpy as np
import pytest

from tensorfem import cook_stress_reference as reference


# A deliberately skew quadrilateral: the Jacobian is strongly non-symmetric, so
# every transpose convention error shows up at full size.
SKEW = np.array(((0.0, 0.0), (10.0, 6.0), (10.0, 9.0), (0.0, 7.0)))
SKEW_Q8 = np.vstack(
    (
        SKEW,
        0.5 * (SKEW[0] + SKEW[1]),
        0.5 * (SKEW[1] + SKEW[2]),
        0.5 * (SKEW[2] + SKEW[3]),
        0.5 * (SKEW[3] + SKEW[0]),
    )
)
AREA = 50.0
GRADIENT = np.array(((1e-3, 2e-3), (-3e-3, 5e-4)))
SHIFT = np.array((0.1, -0.2))


def linear_field(coordinates):
    return np.array([c for point in coordinates for c in SHIFT + GRADIENT @ point])


def exact_strain():
    return np.array(
        (GRADIENT[0, 0], GRADIENT[1, 1], GRADIENT[0, 1] + GRADIENT[1, 0])
    )


def gauss_rule(nodes):
    g = 1.0 / math.sqrt(3.0)
    if nodes == 8:
        points, weights = (-g, g), (1.0, 1.0)
    else:
        root = math.sqrt(3.0 / 5.0)
        points, weights = (-root, 0.0, root), (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)
    return points, weights


@pytest.mark.parametrize(
    "coordinates, function, nodes",
    ((SKEW, reference.q4_strain_matrix, 8), (SKEW_Q8, reference.q8_strain_matrix, 16)),
)
def test_constant_strain_patch_on_skew_element(coordinates, function, nodes):
    """A linear displacement field must be reproduced exactly at every Gauss point."""
    displacement = linear_field(coordinates)
    expected = exact_strain()
    points, _ = gauss_rule(nodes)
    for xi in points:
        for eta in points:
            matrix, measure = function(xi, eta, coordinates)
            assert measure > 0.0
            assert np.allclose(matrix @ displacement, expected, atol=1e-14)


@pytest.mark.parametrize(
    "coordinates, function, nodes",
    ((SKEW, reference.q4_strain_matrix, 8), (SKEW_Q8, reference.q8_strain_matrix, 16)),
)
def test_constant_strain_energy_identity(coordinates, function, nodes):
    """Element strain energy must equal 0.5 * sigma:epsilon * area * t exactly.

    This check uses no convention from the code under test: it multiplies the
    recovered strain by the elasticity matrix and compares the resulting energy
    with the closed-form constant-strain energy.
    """
    elasticity = reference.plane_stress_matrix()
    displacement = linear_field(coordinates)
    strain = exact_strain()
    stress = elasticity @ strain
    exact = 0.5 * strain @ stress * AREA * reference.THICKNESS
    points, weights = gauss_rule(nodes)
    element = np.zeros((nodes, nodes))
    for a, weight_a in zip(points, weights):
        for b, weight_b in zip(points, weights):
            matrix, measure = function(a, b, coordinates)
            element += matrix.T @ elasticity @ matrix * measure * weight_a * weight_b
    computed = 0.5 * displacement @ element @ displacement
    assert abs(computed / exact - 1.0) < 1e-12


@pytest.mark.parametrize(
    "coordinates, function, nodes",
    ((SKEW, reference.q4_strain_matrix, 8), (SKEW_Q8, reference.q8_strain_matrix, 16)),
)
def test_gradient_matches_finite_difference_of_interpolation(coordinates, function, nodes):
    """Recovered gradients must equal a finite difference of the interpolated field."""
    xi = eta = 0.25
    if nodes == 8:
        shape = lambda a, b: 0.25 * np.array(
            ((1 - a) * (1 - b), (1 + a) * (1 - b), (1 + a) * (1 + b), (1 - a) * (1 + b))
        )
    else:
        shape = lambda a, b: reference.q8_shape(a, b)[0]
    displacement = linear_field(coordinates).reshape(-1, 2)

    def physical(point):
        # Newton inversion of the element map from the shape functions themselves.
        a = b = 0.0
        for _ in range(80):
            image = shape(a, b) @ coordinates
            residual = image - point
            h = 1e-7
            jacobian = np.stack(
                (
                    (shape(a + h, b) @ coordinates - shape(a - h, b) @ coordinates) / (2 * h),
                    (shape(a, b + h) @ coordinates - shape(a, b - h) @ coordinates) / (2 * h),
                )
            ).T
            step = np.linalg.solve(jacobian, residual)
            a, b = a - step[0], b - step[1]
            if np.linalg.norm(step) < 1e-14:
                break
        return shape(a, b) @ displacement

    point = shape(xi, eta) @ coordinates
    h = 1e-6
    field_x = (physical(point + [h, 0.0]) - physical(point - [h, 0.0])) / (2 * h)
    field_y = (physical(point + [0.0, h]) - physical(point - [0.0, h])) / (2 * h)
    expected = np.array(
        (
            field_x[0],
            field_y[1],
            field_x[1] + field_y[0],
        )
    )
    matrix, _ = function(xi, eta, coordinates)
    assert np.allclose(matrix @ linear_field(coordinates), expected, atol=1e-7)


def test_untransposed_jacobian_is_rejected_on_skew_element():
    """Guard the transpose convention explicitly: the old form differs materially."""
    xi = eta = 0.25
    _, derivative = reference.q8_shape(xi, eta)
    jacobian = derivative.T @ SKEW_Q8
    correct = derivative @ np.linalg.inv(jacobian).T
    buggy = derivative @ np.linalg.inv(jacobian)
    assert np.abs(buggy - correct).max() > 0.02 * np.abs(correct).max()


def test_q8_reference_reproduces_published_loaded_edge_midpoint():
    """The independent Q8 path must solve the same boundary-value problem as Q4."""
    solution = reference.solve_q8(12)
    probe = reference.probe_displacement(solution)
    assert probe["passed"] is True
    assert probe["relative_error"] < 0.03


def test_loaded_edge_profiles_of_the_two_discretisations_agree():
    """Q4 and Q8 vertical deflections along the loaded edge must agree to 3 %."""
    from tensorfem.additional_benchmark_fields import cook_fields

    fields = cook_fields(32)
    nodes = np.asarray(fields["nodes"])
    displacement = np.asarray(fields["displacement"])
    edge = np.isclose(nodes[:, 0], 48.0)
    order = np.argsort(nodes[edge, 1])
    y_q4 = nodes[edge, 1][order]
    v_q4 = displacement[edge, 1][order]

    solution = reference.solve_q8(12)
    reference_nodes = solution["nodes"]
    reference_edge = np.isclose(reference_nodes[:, 0], 48.0)
    reference_order = np.argsort(reference_nodes[reference_edge, 1])
    y_q8 = reference_nodes[reference_edge, 1][reference_order]
    v_q8 = solution["displacement"].reshape(-1, 2)[reference_edge, 1][reference_order]

    interpolated = np.interp(y_q8, y_q4, v_q4)
    relative = np.abs(interpolated - v_q8) / np.abs(v_q8)
    assert relative.max() < 0.03, relative.max()
    # The profile must also be far from the flat shape the broken oracle produced:
    # an untransposed Jacobian made the Q8 edge run 7.6 -> 51.6 instead of 23.1 -> 25.0.
    assert v_q8[-1] / v_q8[0] < 1.2


def test_platform_q4_pointwise_stress_stays_uncertified():
    """Honest negative result: the element-centre Q4 stress is not yet a 3 % match.

    The audit compares element-centre stress with interior material points, which
    contributes a first-order sampling bias on top of the element error, so the
    margin is still 5 % at n = 64 (see docs/cook-membrane-stress-audit.md).  This
    test fails closed: it must not be relaxed into a pass without a new oracle.
    """
    from tensorfem.additional_benchmark_fields import cook_fields

    fields = cook_fields(16)
    assert fields["stress_accuracy_qualified"] is False
    assert fields["verification"]["passed"] is True  # displacement response only