"""Verifiable sparse structural-grid benchmark utilities.

The model has one scalar displacement per free grid node. Unit springs join
nearest neighbours and boundary springs attach the outer free nodes to a fixed
frame. Its matrix is the standard SPD five-point structural stiffness, not a
diagonal throughput surrogate.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import torch

from .sparse_core import IterativeResult, conjugate_gradient, sparse_mv

Tensor = torch.Tensor


def grid_spring_stiffness(nx: int, ny: int, *, kx: float = 1.0, ky: float = 1.0,
                          dtype=torch.float64, device=None) -> Tensor:
    """Assemble the fixed-boundary rectangular spring grid in sparse COO."""
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be positive")
    if kx <= 0 or ky <= 0:
        raise ValueError("spring stiffnesses must be positive")
    n = nx * ny
    nodes = torch.arange(n, dtype=torch.long, device=device).reshape(ny, nx)
    horizontal_a = nodes[:, :-1].reshape(-1)
    horizontal_b = nodes[:, 1:].reshape(-1)
    vertical_a = nodes[:-1, :].reshape(-1)
    vertical_b = nodes[1:, :].reshape(-1)
    diagonal = torch.full((n,), 2.0*(kx+ky), dtype=dtype, device=device)
    rows = [torch.arange(n, device=device), horizontal_a, horizontal_b,
            vertical_a, vertical_b]
    cols = [torch.arange(n, device=device), horizontal_b, horizontal_a,
            vertical_b, vertical_a]
    values = [diagonal,
              torch.full_like(horizontal_a, -kx, dtype=dtype),
              torch.full_like(horizontal_b, -kx, dtype=dtype),
              torch.full_like(vertical_a, -ky, dtype=dtype),
              torch.full_like(vertical_b, -ky, dtype=dtype)]
    indices = torch.stack((torch.cat(rows), torch.cat(cols)))
    with torch.sparse.check_sparse_tensor_invariants(enable=False):
        return torch.sparse_coo_tensor(indices, torch.cat(values), (n, n),
                                       check_invariants=False).coalesce()


def analytical_mode(nx: int, ny: int, px: int, py: int, *, kx: float = 1.0,
                    ky: float = 1.0, dtype=torch.float64, device=None
                    ) -> tuple[Tensor, float]:
    """Return an exact discrete sine eigenmode and stiffness eigenvalue."""
    if not (1 <= px <= nx and 1 <= py <= ny):
        raise ValueError("mode indices outside the grid")
    x = torch.arange(1, nx+1, dtype=dtype, device=device)
    y = torch.arange(1, ny+1, dtype=dtype, device=device)
    pi = torch.pi
    ux = torch.sin(pi*px*x/(nx+1)); uy = torch.sin(pi*py*y/(ny+1))
    mode = (uy[:, None]*ux[None, :]).reshape(-1)
    eigenvalue = (4*kx*torch.sin(torch.as_tensor(pi*px/(2*(nx+1)), dtype=dtype,
                                                device=device))**2
                  + 4*ky*torch.sin(torch.as_tensor(pi*py/(2*(ny+1)), dtype=dtype,
                                                  device=device))**2)
    return mode, float(eigenvalue)


def manufactured_solution(nx: int, ny: int, *, dtype=torch.float64, device=None
                          ) -> tuple[Tensor, Tensor]:
    """Four-mode exact displacement and load for a nontrivial Krylov solve."""
    # Modes at separated fractions of the spectrum avoid making this a
    # disguised ill-conditioning benchmark; there are four distinct Krylov
    # eigencomponents at every practically relevant size.
    raw = ((max(1, nx//7), max(1, ny//8), 1.0),
           (max(1, nx//4), max(1, ny//3), -.35),
           (max(1, nx//2), max(1, ny//5), .2),
           (max(1, 4*nx//5), max(1, 3*ny//4), -.1))
    choices = tuple(dict(((px, py), (px, py, a)) for px, py, a in raw).values())
    displacement = torch.zeros(nx*ny, dtype=dtype, device=device)
    load = torch.zeros_like(displacement)
    for px, py, amplitude in choices:
        mode, eigenvalue = analytical_mode(nx, ny, px, py, dtype=dtype, device=device)
        displacement += amplitude*mode
        load += amplitude*eigenvalue*mode
    return displacement, load


def sparse_storage_bytes(A: Tensor) -> int:
    """Exact bytes owned by coalesced COO indices and values."""
    if A.layout != torch.sparse_coo or not A.is_coalesced():
        raise ValueError("A must be a coalesced COO tensor")
    return A.indices().numel()*A.indices().element_size() + \
        A.values().numel()*A.values().element_size()


@dataclass(frozen=True)
class GridBenchmarkResult:
    ndof: int
    nnz: int
    storage_bytes: int
    assembly_seconds: float
    solve_seconds: float
    iterations: int
    relative_residual: float
    relative_error: float


def run_grid_benchmark(nx: int, ny: int, *, rtol: float = 1e-9,
                       maxiter: int = 30) -> GridBenchmarkResult:
    """Assemble and solve the analytical multi-mode structural load case."""
    start = time.perf_counter(); A = grid_spring_stiffness(nx, ny)
    assembly = time.perf_counter()-start
    exact, load = manufactured_solution(nx, ny)
    start = time.perf_counter()
    solved: IterativeResult = conjugate_gradient(A, load, rtol=rtol, maxiter=maxiter,
                                                  diagonal=torch.full_like(load, 4.0))
    solve = time.perf_counter()-start
    if not solved.converged:
        raise RuntimeError(f"grid CG failed: residual={solved.relative_residual:g}")
    residual = torch.linalg.vector_norm(sparse_mv(A, solved.x)-load) / \
        torch.linalg.vector_norm(load)
    error = torch.linalg.vector_norm(solved.x-exact) / torch.linalg.vector_norm(exact)
    return GridBenchmarkResult(nx*ny, A._nnz(), sparse_storage_bytes(A), assembly,
                               solve, solved.iterations, float(residual), float(error))
