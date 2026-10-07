import pytest
import torch
import warnings

from tensorfem.sparse_advanced import (gmres, jacobi_inverse,
                                       solve_multiple_rhs, sparse_diagonal)
from tensorfem.sparse_core import conjugate_gradient

D = torch.float64


def test_sparse_jacobi_and_ill_conditioned_benchmark():
    # Condition number 1e8 is deliberately harder than structural smoke tests.
    diagonal = torch.logspace(0, 8, 80, dtype=D)
    A = torch.diag(diagonal).to_sparse_coo()
    b = diagonal.clone()  # analytical solution is one
    inv = jacobi_inverse(A)
    assert torch.equal(sparse_diagonal(A), diagonal)
    result = conjugate_gradient(A, b, diagonal=inv.reciprocal(), rtol=1e-12, maxiter=4)
    assert result.converged
    assert torch.linalg.vector_norm(result.x - 1.) / 80 ** .5 < .03


def test_gmres_symmetric_indefinite_matches_dense_below_three_percent():
    A = torch.tensor([[2., 1., 0., 0.], [1., -1., 1., 0.],
                      [0., 1., 2., 1.], [0., 0., 1., -2.]], dtype=D)
    b = torch.tensor([1., -2., 3., .5], dtype=D)
    result = gmres(A.to_sparse_coo(), b, rtol=1e-12, maxiter=20)
    reference = torch.linalg.solve(A, b)
    assert result.converged
    assert torch.linalg.vector_norm(result.x-reference)/torch.linalg.vector_norm(reference) < .03


def test_csr_backend_with_locally_scoped_pytorch_beta_notice():
    A = torch.tensor([[4., -1.], [-1., 3.]], dtype=D)
    # PyTorch 2.12 emits an upstream CSR-beta UserWarning on first construction.
    # Scope suppression to that exact message; all solver warnings remain errors.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state.*",
                                category=UserWarning)
        csr = A.to_sparse_csr()
    assert torch.equal(sparse_diagonal(csr), torch.diagonal(A))
    b = torch.tensor([2., 5.], dtype=D)
    result = gmres(csr, b, rtol=1e-12)
    assert result.converged
    assert torch.allclose(result.x, torch.linalg.solve(A, b), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("method", ["cg", "gmres"])
def test_multiple_rhs_matches_dense(method):
    A = torch.tensor([[5., -1., 0.], [-1., 4., -1.], [0., -1., 3.]], dtype=D)
    B = torch.tensor([[1., 0., 2.], [0., 3., -1.], [2., 1., 0.]], dtype=D)
    inv = jacobi_inverse(A)
    result = solve_multiple_rhs(A.to_sparse(), B, method=method, jacobi=inv,
                                rtol=1e-12, maxiter=30)
    assert all(result.converged)
    assert torch.allclose(result.x, torch.linalg.solve(A, B), rtol=1e-9, atol=1e-11)


def test_jacobi_fails_closed_and_cuda_consistency_if_available():
    with pytest.raises(ValueError, match="zero"):
        jacobi_inverse(torch.tensor([[0., 1.], [1., 2.]], dtype=D).to_sparse())
    if torch.cuda.is_available():
        A = torch.tensor([[3., 1.], [1., -2.]], dtype=D)
        b = torch.tensor([2., 5.], dtype=D)
        cpu = gmres(A.to_sparse(), b, rtol=1e-12)
        gpu = gmres(A.cuda().to_sparse(), b.cuda(), rtol=1e-12)
        assert gpu.x.is_cuda
        assert torch.allclose(cpu.x, gpu.x.cpu(), rtol=1e-9, atol=1e-11)
