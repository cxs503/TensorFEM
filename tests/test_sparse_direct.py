import pytest
import torch

from tensorfem.sparse_direct import (
    SparseBackendUnavailable, factorize_sparse, scipy_sparse_status,
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
