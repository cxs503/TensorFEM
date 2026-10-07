import pytest
import torch

from tensorfem.sparse_core import sparse_mv
from tensorfem.structural_grid_benchmark import (
    analytical_mode, grid_spring_stiffness, run_grid_benchmark,
    sparse_storage_bytes,
)

D = torch.float64


def test_grid_assembly_is_coupled_spd_and_matches_dense_reference():
    A = grid_spring_stiffness(5, 4)
    dense = A.to_dense()
    assert A._nnz() == 5*4 + 2*((5-1)*4 + 5*(4-1))
    assert torch.count_nonzero(dense-torch.diag(torch.diagonal(dense))) > 0
    assert torch.linalg.cholesky_ex(dense).info.item() == 0
    load = torch.arange(1, 21, dtype=D)
    dense_solution = torch.linalg.solve(dense, load)
    assert torch.linalg.vector_norm(dense@dense_solution-load) < 1e-12


@pytest.mark.parametrize("px,py", [(1,1), (2,3), (5,4)])
def test_discrete_analytical_eigenmodes(px, py):
    A = grid_spring_stiffness(7, 6)
    mode, eigenvalue = analytical_mode(7, 6, px, py)
    error = torch.linalg.vector_norm(sparse_mv(A, mode)-eigenvalue*mode) / \
        torch.linalg.vector_norm(eigenvalue*mode)
    assert float(error) < 1e-13


def test_ci_scale_solve_has_strict_error_and_residual_gate():
    result = run_grid_benchmark(40, 32)
    assert result.ndof == 1280 and result.nnz > result.ndof
    assert result.iterations <= 8
    assert result.relative_residual < 1e-9
    assert result.relative_error < 1e-8  # stricter than the project 3% gate
    assert result.storage_bytes > result.nnz*8


def test_invalid_inputs_fail_closed():
    with pytest.raises(ValueError, match="positive"):
        grid_spring_stiffness(0, 2)
    with pytest.raises(ValueError, match="positive"):
        grid_spring_stiffness(2, 2, kx=0)
    with pytest.raises(ValueError, match="outside"):
        analytical_mode(2, 2, 3, 1)
    with pytest.raises(ValueError, match="coalesced COO"):
        sparse_storage_bytes(torch.eye(2))
