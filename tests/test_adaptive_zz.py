"""Acceptance tests for the ZZ indicator and the indicator-driven h-refinement loop.

Anchors
-------
* **Exact solution**: the Williams re-entrant-corner eigen-solution is an exact solution of
  the homogeneous equilibrium problem with traction-free notch faces, and its displacement
  field is checked against the strain-stress relation by finite differences.
* **Exact energy**: the closed-form strain energy of that field is quadrature-independent.
* **Recovery operator**: the superconvergent patch recovery reproduces a linear stress field
  exactly (a recovery that does not would inject energy into the indicator).
* **Conformity**: a mesh with hanging nodes still passes the linear patch test, i.e. the
  multipoint constraints really do span a conforming subspace.
* **Estimator quality**: the effectivity index is close to one and the estimated error
  tracks the independently computed true error.
* **Adaptivity**: two Doerfler-marked refinement rounds reach a lower error than a uniform
  mesh with 2x the unknowns, and the marked set captures the error energy it was asked for.
* **Fail closed**: a negative / non-finite indicator, a singular compliance matrix and a
  malformed marking fraction are rejected instead of being silently clipped.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfem.adaptive_zz import (
    CORNER_ANGLE,
    WILLIAMS_LAMBDA,
    QuadMesh,
    boundary_nodes,
    constraint_pairs,
    dorfler_mark,
    elasticity_matrix,
    effectivity_index,
    element_indicators,
    exact_dirichlet,
    exact_energy_norm_sq,
    gauss_point_stresses,
    hanging_nodes,
    lshape_mesh,
    refine_mesh,
    solve_displacement,
    solve_level,
    spr_recover,
    study,
    true_element_error_sq,
    validate_indicators,
    williams_displacement,
    williams_exponent,
    williams_strain,
    williams_stress,
)


# ---------------------------------------------------------------------------
# 1. the exact solution itself
# ---------------------------------------------------------------------------
def test_williams_exponent_matches_reference_value():
    assert williams_exponent() == pytest.approx(WILLIAMS_LAMBDA, abs=1e-12)
    # a wider opening angle weakens the singularity: the crack limit (2 pi) gives 1/2
    assert williams_exponent(2.0 * math.pi) == pytest.approx(0.5, abs=1e-3)
    assert williams_exponent(1.9 * math.pi) < WILLIAMS_LAMBDA
    with pytest.raises(ValueError):
        williams_exponent(CORNER_ANGLE, lower=2.0, upper=1.0)


def test_notch_faces_are_traction_free_and_equilibrium_holds():
    lam = williams_exponent()
    x = -np.linspace(0.05, 0.95, 7)  # notch face y = 0, x < 0
    a = williams_stress(np.column_stack([x, np.zeros_like(x)]), lam=lam)
    traction_a = np.column_stack([-a[:, 2], -a[:, 1]])  # outward normal (0, -1)
    y = -np.linspace(0.05, 0.95, 7)  # notch face x = 0, y < 0
    b = williams_stress(np.column_stack([np.zeros_like(y), y]), lam=lam)
    traction_b = np.column_stack([b[:, 0], b[:, 2]])  # outward normal (1, 0)
    scale = np.abs(williams_stress(np.array([[0.05, 0.05]]), lam=lam)).max()
    assert np.abs(traction_a).max() < 1e-12 * scale
    assert np.abs(traction_b).max() < 1e-12 * scale

    points = np.array([[0.3, 0.4], [-0.6, 0.2], [-0.5, -0.4], [0.8, -0.3], [-0.9, 0.7]])
    step = 1e-6
    d_dx = (
        williams_stress(points + [step, 0.0], lam=lam)
        - williams_stress(points - [step, 0.0], lam=lam)
    ) / (2 * step)
    d_dy = (
        williams_stress(points + [0.0, step], lam=lam)
        - williams_stress(points - [0.0, step], lam=lam)
    ) / (2 * step)
    divergence_x = np.abs(d_dx[:, 0] + d_dy[:, 2]).max()
    divergence_y = np.abs(d_dx[:, 2] + d_dy[:, 1]).max()
    assert max(divergence_x, divergence_y) < 1e-6 * np.abs(williams_stress(points, lam=lam)).max()


def test_exact_displacement_reproduces_the_strain_field():
    lam = williams_exponent()
    points = np.array([[0.4, 0.1], [-0.5, 0.3], [-0.7, -0.2], [0.9, 0.6]])
    step = 1e-6
    du_dx = (
        williams_displacement(points + [step, 0.0], lam=lam)
        - williams_displacement(points - [step, 0.0], lam=lam)
    ) / (2 * step)
    du_dy = (
        williams_displacement(points + [0.0, step], lam=lam)
        - williams_displacement(points - [0.0, step], lam=lam)
    ) / (2 * step)
    strain = np.column_stack([du_dx[:, 0], du_dy[:, 1], du_dx[:, 1] + du_dy[:, 0]])
    reference = williams_strain(points, lam=lam)
    assert np.linalg.norm(strain - reference) < 1e-7 * np.linalg.norm(reference)


def test_exact_energy_is_independent_of_the_angular_quadrature():
    coarse = exact_energy_norm_sq(panels=128, order=10)
    fine = exact_energy_norm_sq(panels=512, order=12)
    assert fine > 0.0
    assert coarse == pytest.approx(fine, rel=1e-10)


# ---------------------------------------------------------------------------
# 2. recovery operator and conforming refinement
# ---------------------------------------------------------------------------
def linear_stress(points: np.ndarray) -> np.ndarray:
    points = np.atleast_2d(points)
    return np.stack(
        [
            1.0 + 2.0 * points[:, 0] - 0.5 * points[:, 1],
            -3.0 + 0.25 * points[:, 0] + 1.5 * points[:, 1],
            0.7 - 0.9 * points[:, 0] + 0.2 * points[:, 1],
        ],
        axis=1,
    )


def test_patch_recovery_is_exact_for_a_linear_stress_field():
    mesh = lshape_mesh(8)
    _, gauss_points = gauss_point_stresses(mesh, np.zeros(2 * mesh.n_nodes), elasticity_matrix())
    planted = linear_stress(gauss_points.reshape(-1, 2)).reshape(gauss_points.shape[:2] + (3,))
    nodal = spr_recover(mesh, planted, gauss_points)
    error = np.abs(nodal - linear_stress(mesh.nodes)).max()
    assert error < 1e-9 * np.abs(linear_stress(mesh.nodes)).max()


def test_mesh_with_hanging_nodes_passes_the_linear_patch_test():
    mesh = lshape_mesh(8)
    marked = np.zeros(mesh.n_elements, dtype=bool)
    marked[[0, 7, 8, 9]] = True  # refine a clustered subset -> hanging nodes appear
    refined = refine_mesh(mesh, marked)
    assert len(hanging_nodes(refined)) > 0, "the test mesh must actually have hanging nodes"

    pairs = constraint_pairs(refined)
    assert pairs, "hanging nodes must produce constraints"
    for _, masters in pairs:
        assert sum(masters.values()) == pytest.approx(1.0)
        assert all(coefficient > 0.0 for coefficient in masters.values())
        assert len(masters) >= 2, "a hanging value never depends on a single master"

    # constant-strain (linear displacement) patch: exact field, zero interior error
    slope = np.array([[2.0e-3, -5.0e-4], [3.0e-4, 1.0e-3]])
    offset = np.array([1.0e-3, -2.0e-3])

    def linear_displacement(points: np.ndarray) -> np.ndarray:
        return np.atleast_2d(points) @ slope.T + offset

    nodes = boundary_nodes(refined)
    boundary_values = linear_displacement(refined.nodes[nodes])
    prescribed: dict[int, float] = {}
    for k, node in enumerate(nodes):
        prescribed[2 * int(node)] = float(boundary_values[k, 0])
        prescribed[2 * int(node) + 1] = float(boundary_values[k, 1])
    displacement, _ = solve_displacement(
        refined, elasticity_matrix(), dirichlet=prescribed, rtol=1e-13
    )
    recovered = displacement.reshape(-1, 2)
    assert np.abs(recovered - linear_displacement(refined.nodes)).max() < 1e-10


# ---------------------------------------------------------------------------
# 3. estimator quality and the adaptive gain
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_study():
    return study(
        divisions=8, rounds=2, fraction=0.5, uniform_divisions=(8, 12, 16, 24), max_subcells=6
    )


def test_effectivity_index_is_close_to_one_and_error_decreases(small_study):
    uniform = small_study["uniform"]["levels"]
    errors = [level["true_energy_norm"] for level in uniform]
    assert all(later < earlier for earlier, later in zip(errors, errors[1:])), errors
    for level in uniform:
        assert 0.85 < level["effectivity"] < 1.2, level["effectivity"]
    # estimated and true errors must move together, not just be small
    estimated = [level["estimated_energy_norm"] for level in uniform]
    assert all(later < earlier for earlier, later in zip(estimated, estimated[1:]))
    # the recovery becomes more accurate away from the corner as the mesh is refined
    smooth = [level["recovery_nodal_error_smooth"] for level in uniform]
    assert smooth[-1] < smooth[0]


def test_two_marked_rounds_beat_uniform_at_matched_dofs(small_study):
    levels = small_study["adaptive"]["levels"]
    assert len(levels) == 3, "the loop must record the start mesh plus two refinement rounds"
    assert len(small_study["adaptive"]["markings"]) == 2
    assert levels[-1]["hanging_nodes"] > 0, "refinement must actually create hanging nodes"
    assert levels[-1]["true_energy_norm"] < levels[0]["true_energy_norm"]

    matched = [row for row in small_study["matched_dof_table"] if "uniform_error" in row]
    assert matched, "no uniform level reaches the adaptive DOF count"
    final = matched[-1]
    assert final["uniform_dofs"] >= final["adaptive_dofs"]
    assert final["uniform_over_adaptive_error_ratio"] > 1.05, final
    equal = small_study["equal_error_table"][-1]
    assert equal["uniform_dofs_ratio"] > 1.5, equal


def test_dorfler_marking_captures_the_requested_error_energy():
    mesh = lshape_mesh(8)
    result = solve_level(mesh, max_subcells=6)
    marked = dorfler_mark(result.indicators, 0.5)
    captured = (result.indicators[marked] ** 2).sum() / (result.indicators**2).sum()
    assert captured >= 0.5
    assert marked.sum() < result.indicators.size, "marking everything is not bulk marking"
    full = dorfler_mark(result.indicators, 1.0)
    assert full.all()


def test_true_error_quadrature_is_converged():
    mesh = lshape_mesh(8)
    prescribed = exact_dirichlet(mesh)
    displacement, _ = solve_displacement(mesh, elasticity_matrix(), dirichlet=prescribed)
    coarse = np.sqrt(
        true_element_error_sq(mesh, displacement, elasticity_matrix(), max_subcells=8).sum()
    )
    fine = np.sqrt(
        true_element_error_sq(mesh, displacement, elasticity_matrix(), max_subcells=16).sum()
    )
    assert coarse == pytest.approx(fine, rel=1e-4)
    assert fine > 0.0


# ---------------------------------------------------------------------------
# 4. fail-closed / negative cases
# ---------------------------------------------------------------------------
def test_negative_and_non_finite_indicators_are_rejected():
    with pytest.raises(ValueError, match="negative"):
        validate_indicators(np.array([1.0, -0.5, 2.0]))
    with pytest.raises(ValueError, match="non-finite"):
        validate_indicators(np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match="one-dimensional"):
        validate_indicators(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="empty"):
        validate_indicators(np.array([]))
    with pytest.raises(ValueError):
        dorfler_mark(np.array([0.0, 0.0]))  # nothing to mark
    with pytest.raises(ValueError):
        dorfler_mark(np.array([1.0, 2.0]), 0.0)  # invalid fraction
    with pytest.raises(ValueError):
        effectivity_index(1.0, 0.0)  # a zero exact norm cannot be scored


def test_indicator_rejects_a_non_positive_energy_density():
    mesh = lshape_mesh(4)
    stress = np.zeros((mesh.n_elements, 4, 3))
    recovered = np.zeros((mesh.n_elements, 4, 3))
    good = elasticity_matrix()
    with pytest.raises(ValueError, match="symmetric"):
        element_indicators(mesh, stress, recovered, good + np.triu(np.ones((3, 3))))
    indefinite = good.copy()
    indefinite[0, 0] = -abs(good[0, 0])
    with pytest.raises(ValueError, match="positive definite"):
        element_indicators(mesh, stress, recovered, indefinite)
    zero = element_indicators(mesh, stress, recovered, good)
    assert np.all(zero == 0.0)


def test_mesh_helpers_reject_malformed_input():
    with pytest.raises(ValueError):
        lshape_mesh(7)  # odd divisions cannot resolve the notch
    mesh = lshape_mesh(4)
    with pytest.raises(ValueError):
        refine_mesh(mesh, np.ones(mesh.n_elements + 1, dtype=bool))
    with pytest.raises(ValueError):
        QuadMesh(mesh.nodes, mesh.elements[:, :3])
