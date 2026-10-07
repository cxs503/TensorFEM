"""Portable advanced iterative solvers built only from public PyTorch APIs.

This module intentionally remains independent of the package entry points so
that applications can qualify it before adopting it.  It supports dense, COO,
CSR and matrix-free operators on CPU or CUDA.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .sparse_core import IterativeResult, conjugate_gradient, sparse_mv

Tensor = torch.Tensor
Operator = Tensor | Callable[[Tensor], Tensor]


def sparse_diagonal(A: Tensor) -> Tensor:
    """Extract a matrix diagonal without densifying COO/CSR tensors."""
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if A.layout == torch.sparse_csr:
        A = A.to_sparse_coo()
    if A.layout == torch.sparse_coo:
        A = A.coalesce()
        i = A.indices()
        mask = i[0] == i[1]
        out = torch.zeros(A.shape[0], dtype=A.dtype, device=A.device)
        return out.scatter_add(0, i[0, mask], A.values()[mask])
    return torch.diagonal(A)


def jacobi_inverse(A: Tensor, *, absolute: bool = False,
                   min_diagonal: float | None = None) -> Tensor:
    """Return the inverse Jacobi diagonal, failing on singular entries.

    ``absolute=True`` is useful as a positive preconditioner for indefinite
    systems; it does not alter the original operator.
    """
    diagonal = sparse_diagonal(A)
    if absolute:
        diagonal = diagonal.abs()
    scale = float(diagonal.abs().max()) if diagonal.numel() else 0.0
    floor = min_diagonal if min_diagonal is not None else torch.finfo(diagonal.dtype).eps * max(scale, 1.0)
    if bool(torch.any(diagonal.abs() <= floor)):
        raise ValueError("Jacobi preconditioner has a zero or near-zero diagonal")
    return diagonal.reciprocal()


def _apply(operator: Operator, x: Tensor) -> Tensor:
    return operator(x) if callable(operator) else sparse_mv(operator, x)


def gmres(A: Operator, b: Tensor, *, x0: Tensor | None = None,
          rtol: float = 1e-10, atol: float = 0.0, maxiter: int | None = None,
          restart: int | None = None,
          inverse_preconditioner: Tensor | Callable[[Tensor], Tensor] | None = None
          ) -> IterativeResult:
    """Restarted left-preconditioned GMRES for general real systems.

    The reported and stopping residual is always the true, unpreconditioned
    residual.  Arnoldi least-squares use ``torch.linalg.lstsq`` and therefore
    run on the same CPU/CUDA device as the input.
    """
    if b.ndim != 1:
        raise ValueError("b must be one-dimensional")
    if not b.is_floating_point():
        raise TypeError("b must have floating dtype")
    n = b.numel()
    maxiter = maxiter or max(10, 2 * n)
    restart = min(restart or min(40, n), n, maxiter)
    if restart < 1:
        raise ValueError("restart must be positive")
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    if x.shape != b.shape:
        raise ValueError("x0 and b sizes differ")
    if inverse_preconditioner is None:
        pre = lambda v: v
    elif callable(inverse_preconditioner):
        pre = inverse_preconditioner
    else:
        if inverse_preconditioner.shape != b.shape:
            raise ValueError("inverse preconditioner and b sizes differ")
        pre = lambda v: inverse_preconditioner * v
    norm_b = float(torch.linalg.vector_norm(b))
    threshold = max(float(atol), float(rtol) * norm_b)
    iterations = 0
    while iterations < maxiter:
        true_r = b - _apply(A, x)
        true_norm = float(torch.linalg.vector_norm(true_r))
        if true_norm <= threshold:
            return IterativeResult(x, True, iterations, true_norm / max(norm_b, 1e-300))
        r = pre(true_r)
        beta = torch.linalg.vector_norm(r)
        if float(beta) == 0.0:
            return IterativeResult(x, False, iterations, true_norm / max(norm_b, 1e-300))
        width = min(restart, maxiter - iterations)
        V = torch.zeros((n, width + 1), dtype=b.dtype, device=b.device)
        H = torch.zeros((width + 1, width), dtype=b.dtype, device=b.device)
        V[:, 0] = r / beta
        base = x.clone()
        for j in range(width):
            w = pre(_apply(A, V[:, j]))
            # Modified Gram-Schmidt is more stable than a single projection.
            for i in range(j + 1):
                H[i, j] = torch.dot(V[:, i], w)
                w = w - H[i, j] * V[:, i]
            H[j + 1, j] = torch.linalg.vector_norm(w)
            if float(H[j + 1, j]) > torch.finfo(b.dtype).eps:
                V[:, j + 1] = w / H[j + 1, j]
            rhs = torch.zeros(j + 2, dtype=b.dtype, device=b.device)
            rhs[0] = beta
            y = torch.linalg.lstsq(H[:j + 2, :j + 1], rhs).solution
            x = base + V[:, :j + 1] @ y
            iterations += 1
            true_norm = float(torch.linalg.vector_norm(b - _apply(A, x)))
            if true_norm <= threshold:
                return IterativeResult(x, True, iterations, true_norm / max(norm_b, 1e-300))
            if float(H[j + 1, j]) <= torch.finfo(b.dtype).eps:
                break
    true_norm = float(torch.linalg.vector_norm(b - _apply(A, x)))
    return IterativeResult(x, False, iterations, true_norm / max(norm_b, 1e-300))


@dataclass(frozen=True)
class MultiRHSResult:
    x: Tensor
    converged: tuple[bool, ...]
    iterations: tuple[int, ...]
    relative_residuals: tuple[float, ...]


def solve_multiple_rhs(A: Operator, B: Tensor, *, method: str = "cg",
                       rtol: float = 1e-10, atol: float = 0.0,
                       maxiter: int | None = None, restart: int | None = None,
                       jacobi: Tensor | None = None) -> MultiRHSResult:
    """Solve ``A X=B`` for many load cases without duplicating matrix storage."""
    if B.ndim != 2:
        raise ValueError("B must have shape (ndof, nrhs)")
    results: list[IterativeResult] = []
    for j in range(B.shape[1]):
        if method == "cg":
            diagonal = None if jacobi is None else jacobi.reciprocal()
            result = conjugate_gradient(A, B[:, j], rtol=rtol, atol=atol,
                                        maxiter=maxiter, diagonal=diagonal)
        elif method == "gmres":
            result = gmres(A, B[:, j], rtol=rtol, atol=atol, maxiter=maxiter,
                           restart=restart, inverse_preconditioner=jacobi)
        else:
            raise ValueError("method must be 'cg' or 'gmres'")
        results.append(result)
    return MultiRHSResult(torch.stack([r.x for r in results], 1),
                          tuple(r.converged for r in results),
                          tuple(r.iterations for r in results),
                          tuple(r.relative_residual for r in results))
