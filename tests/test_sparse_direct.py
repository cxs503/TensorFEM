import pytest
import torch

from tensorfem.sparse_direct import (
    SparseBackendUnavailable, SparseLinearSolver, factorize_sparse,
    scipy_sparse_status, sparse_backend_statuses,
)


def test_optional_backend_probe_and_fail_closed_contract():
    status = scipy_sparse_status()
    assert status.name == "scipy"
    matrix = torch.eye(2, dtype=torch.float64).to_sparse_coo()
    if not status.available:
        with pytest.raises(SparseBackendUnavailable):
            factorize_sparse(matrix)


def test_sparse_direct_matches_dense_when_scipy_is_available():
    if not scipy_sparse_status().available:
        pytest.skip("optional SciPy backend is absent")
    dense = torch.tensor([[4., 1., 0.], [1., 3., 1.], [0., 1., 2.]],
                         dtype=torch.float64)
    rhs = torch.tensor([1., 2., 3.], dtype=torch.float64)
    direct = factorize_sparse(dense.to_sparse_coo(), method="splu").solve(rhs)
    assert torch.allclose(direct, torch.linalg.solve(dense, rhs),
                          rtol=2e-13, atol=2e-13)
    ilu = factorize_sparse(dense.to_sparse_coo(), method="spilu",
                           drop_tolerance=0., fill_factor=20.).solve(rhs)
    assert torch.allclose(ilu, direct, rtol=2e-13, atol=2e-13)


def test_sparse_direct_rejects_dense_and_unknown_method():
    with pytest.raises(ValueError, match="method"):
        factorize_sparse(torch.eye(2).to_sparse_coo(), method="other")
    if scipy_sparse_status().available:
        with pytest.raises(TypeError, match="sparse"):
            factorize_sparse(torch.eye(2))


def test_sparse_direct_singular_factorization_fails_closed():
    if not scipy_sparse_status().available:
        pytest.skip("optional SciPy backend is absent")
    singular = torch.tensor([[1., 1.], [1., 1.]], dtype=torch.float64)
    with pytest.raises(RuntimeError, match="factorization failed"):
        factorize_sparse(singular.to_sparse_coo())


def test_sparse_direct_rejects_nonfinite_inputs_when_backend_is_available():
    if not scipy_sparse_status().available:
        pytest.skip("optional SciPy backend is absent")
    matrix = torch.tensor([[float("nan"), 0.], [0., 1.]], dtype=torch.float64)
    with pytest.raises(ValueError, match="non-finite"):
        factorize_sparse(matrix.to_sparse_coo())

    factor = factorize_sparse(torch.eye(2, dtype=torch.float64).to_sparse_coo())
    with pytest.raises(ValueError, match="non-finite"):
        factor.solve(torch.tensor([1., float("inf")], dtype=torch.float64))


def test_unified_solver_reuses_factorization_for_vector_and_multiple_rhs():
    if not scipy_sparse_status().available:
        pytest.skip("optional SciPy backend is absent")
    dense = torch.tensor([[5., 1., 0.], [1., 4., 1.], [0., 1., 3.]],
                         dtype=torch.float64)
    factor = SparseLinearSolver("auto").factorize(dense.to_sparse_coo())
    vector = torch.tensor([1., 2., 3.], dtype=torch.float64)
    multiple = torch.stack((vector, 2*vector), dim=1)
    assert torch.allclose(factor.solve(vector), torch.linalg.solve(dense, vector),
                          rtol=2e-13, atol=2e-13)
    assert torch.allclose(factor.solve(multiple), torch.linalg.solve(dense, multiple),
                          rtol=2e-13, atol=2e-13)
    diagnostics = factor.diagnostics
    assert diagnostics.solve_calls == 2
    assert diagnostics.right_hand_sides == 3
    assert diagnostics.factorization_seconds >= 0.
    assert diagnostics.solve_seconds >= 0.
    assert diagnostics.estimated_factor_bytes > 0
    assert diagnostics.last_relative_residual is not None
    assert diagnostics.last_relative_residual < 1e-13


def test_unavailable_unified_backends_fail_closed():
    statuses = {row.name: row for row in sparse_backend_statuses()}
    assert "superlu" in statuses and "petsc" in statuses and "mumps" in statuses
    for name in ("petsc", "mumps", "pardiso", "suitesparse", "amg"):
        if not statuses[name].available:
            with pytest.raises(SparseBackendUnavailable):
                SparseLinearSolver(name)
    with pytest.raises(SparseBackendUnavailable, match="not implemented"):
        SparseLinearSolver("unknown")
