"""Acceptance tests for the 2D Q4 analytic-order audit of the recovered-gradient indicator.

Anchors
-------
* **Implementation cross-check**: the vectorised 1D-difference gradient of a rectangular element
  is compared with the per-element ``J^-1 dN/dxi`` formulation, so the speed-up cannot hide a
  factor error.
* **Assembly invariants**: the stiffness is symmetric and annihilates constant fields, and the
  load integrates to zero because the exact flux vanishes on the boundary.
* **Recovery exactness**: nodal averaging reproduces a linear field at the nodes.
* **Orders**: energy ``O(N^-1/2)`` and L2 ``O(N^-1)`` against theory, measured on a ladder whose
  integrals are exact for the polynomial manufactured solution.
* **Effectivity**: ``theta -> 1`` (asymptotic exactness of the indicator) on the smooth field.
* **Recovery value**: the recovered gradient is closer to the exact gradient than the raw element
  gradient, which is the property the indicator relies on.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorfem.adaptive_poisson_2d import (
    THEORY_RATES,
    BubbleSolution,
    Mesh2D,
    assemble,
    asymptotic_rate,
    gradient_from_nodal,
    indicator_norms,
    interpolate_nodal,
    recover_nodal_gradient,
    shape_gradients,
    solve,
    element_jacobian,
    study,
    uniform_study,
)


@pytest.fixture(scope="module")
def ladder():
    return uniform_study(BubbleSolution(), (4, 8, 16, 24))


@pytest.fixture(scope="module")
def payload():
    return study(divisions=(4, 8, 16))


def test_vectorised_gradient_matches_the_per_element_formulation():
    mesh = Mesh2D(3, 2)
    rng = np.random.default_rng(20261106)
    nodal = rng.normal(size=mesh.nnodes)
    coords = np.column_stack(
        (
            np.tile(np.linspace(0.0, 1.0, mesh.nx + 1), mesh.ny + 1),
            np.repeat(np.linspace(0.0, 1.0, mesh.ny + 1), mesh.nx + 1),
        )
    )
    for xi, eta in ((0.0, 0.0), (0.5773502691896257, -0.5773502691896257)):
        fast_x, fast_y = gradient_from_nodal(mesh, nodal, xi, eta)
        for e, quad in enumerate(mesh.elements()):
            jacobian = coords[quad].T @ shape_gradients(xi, eta).T
            reference = np.linalg.solve(jacobian, shape_gradients(xi, eta) @ nodal[quad])
            assert fast_x.ravel()[e] == pytest.approx(reference[0], rel=1e-12, abs=1e-12)
            assert fast_y.ravel()[e] == pytest.approx(reference[1], rel=1e-12, abs=1e-12)


def test_assembly_is_symmetric_and_load_is_balanced():
    mesh = Mesh2D(4, 3)
    solution = BubbleSolution()
    K, F = assemble(mesh, solution)
    assert np.allclose(K, K.T, atol=0.0)
    # a bilinear field is reproduced exactly, so K annihilates constants
    assert np.abs(K.sum(axis=1)).max() < 1e-12
    # the exact flux vanishes on the whole boundary, so the total load must vanish
    assert F.sum() == pytest.approx(0.0, abs=1e-12)


def test_manufactured_solution_has_unit_peak_and_zero_trace():
    solution = BubbleSolution()
    assert float(solution.u(np.array([0.5]), np.array([0.5]))[0]) == pytest.approx(1.0)
    interior = np.array([0.25, 0.4, 0.75])
    for edge in (0.0, 1.0):
        assert np.allclose(solution.u(np.full_like(interior, edge), interior), 0.0, atol=0.0)
        assert np.allclose(solution.u(interior, np.full_like(interior, edge)), 0.0, atol=0.0)
    gx, gy = solution.gradient(np.array([0.0, 1.0]), np.array([0.3, 0.7]))
    assert np.allclose(gx, 0.0, atol=0.0) and np.allclose(gy, 0.0, atol=0.0)


def test_recovery_reproduces_a_linear_field_at_the_nodes():
    mesh = Mesh2D(4, 3)
    x = np.tile(np.linspace(0.0, 1.0, mesh.nx + 1), mesh.ny + 1)
    y = np.repeat(np.linspace(0.0, 1.0, mesh.ny + 1), mesh.nx + 1)
    nodal = 2.0 * x - 5.0 * y
    recovered = recover_nodal_gradient(mesh, nodal)
    assert np.allclose(recovered[:, 0], 2.0, atol=1e-12)
    assert np.allclose(recovered[:, 1], -5.0, atol=1e-12)
    # interpolation of a constant field is that constant everywhere
    assert np.allclose(interpolate_nodal(mesh, recovered[:, 0], 0.3, -0.2), 2.0, atol=1e-12)


def test_energy_and_l2_orders_match_theory(ladder):
    records = ladder["records"]
    dofs = [record["dofs"] for record in records]
    energy_rate = asymptotic_rate(dofs, [record["true_energy_norm"] for record in records])
    l2_rate = asymptotic_rate(dofs, [record["l2_error"] for record in records])
    assert energy_rate == pytest.approx(THEORY_RATES["energy"], abs=0.08), energy_rate
    assert l2_rate == pytest.approx(THEORY_RATES["l2"], abs=0.12), l2_rate
    errors = [record["true_energy_norm"] for record in records]
    assert all(later < earlier for earlier, later in zip(errors, errors[1:]))


def test_effectivity_tends_to_one(ladder):
    effectivity = [record["effectivity"] for record in ladder["records"]]
    assert all(value > 0.8 for value in effectivity)
    assert effectivity[0] == pytest.approx(1.0, abs=0.1)
    assert effectivity[-1] == pytest.approx(1.0, abs=0.01)


def test_recovered_gradient_is_closer_to_the_truth_than_the_raw_gradient(ladder):
    records = ladder["records"]
    # from the second rung onwards the recovered gradient is the better field
    for record in records[2:]:
        assert record["recovered_gradient_l2_error"] < record["true_energy_norm"]
    # and it converges at least as fast as the energy norm
    dofs = [record["dofs"] for record in records]
    recovered_rate = asymptotic_rate(
        dofs, [record["recovered_gradient_l2_error"] for record in records]
    )
    energy_rate = asymptotic_rate(dofs, [record["true_energy_norm"] for record in records])
    assert recovered_rate < energy_rate


def test_indicator_is_non_negative_and_elementwise(payload):
    mesh = Mesh2D(4, 4)
    solution = BubbleSolution()
    K, F = assemble(mesh, solution)
    u = solve(mesh, K, F)
    eta = indicator_norms(mesh, u)
    assert eta.shape == (mesh.nelements,)
    assert np.all(np.isfinite(eta)) and np.all(eta >= 0.0)
    assert float(np.sqrt(np.sum(eta**2))) > 0.0
    # the indicator must vanish where the field is exactly reproduced: a linear field
    x = np.tile(np.linspace(0.0, 1.0, mesh.nx + 1), mesh.ny + 1)
    linear = indicator_norms(mesh, 3.0 * x)
    assert np.allclose(linear, 0.0, atol=1e-12)


def test_payload_records_the_order_table_and_criteria(payload):
    assert payload["schema"] == "tensorfem.adaptive_poisson_2d.study/v1"
    assert set(payload["rates"]) == {"energy", "l2", "recovered_gradient"}
    assert payload["criteria"]["final_effectivity"] == pytest.approx(1.0, abs=0.05)
    assert payload["criteria"]["final_dofs"] > 0
    assert payload["criteria"]["final_seconds"] >= 0.0
    assert len(payload["levels"]) == 3


def test_fail_closed_on_malformed_meshes():
    with pytest.raises(ValueError):
        Mesh2D(0, 3)
    with pytest.raises(ValueError):
        Mesh2D(3, -1)
    mesh = Mesh2D(2, 2)
    with pytest.raises(ValueError):
        mesh.grid(np.zeros(3))
    # a degenerate (zero-area) element must be rejected by the Jacobian gate
    with pytest.raises(ValueError):
        element_jacobian(
            np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]), 0.0, 0.0
        )