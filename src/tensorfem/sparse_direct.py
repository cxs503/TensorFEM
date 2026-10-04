"""Optional CPU sparse direct and ILU adapters.

SciPy is deliberately imported only when a factorization is requested.  The
core TensorFEM installation therefore has no NumPy/SciPy dependency.  Missing
or unsupported backends fail closed instead of silently densifying a matrix or
changing the requested algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Callable

import torch


class SparseBackendUnavailable(RuntimeError):
    """Raised when an explicitly requested optional backend cannot be used."""


@dataclass(frozen=True)
class SparseBackendStatus:
    name: str
    available: bool
    reason: str


def scipy_sparse_status() -> SparseBackendStatus:
    """Report availability without importing an optional numerical stack."""
    if find_spec("numpy") is None:
        return SparseBackendStatus("scipy", False, "NumPy is not installed")
    if find_spec("scipy") is None:
        return SparseBackendStatus("scipy", False, "SciPy is not installed")
    return SparseBackendStatus("scipy", True, "available")


def _scipy_modules():
    status = scipy_sparse_status()
    if not status.available:
        raise SparseBackendUnavailable(status.reason)
    try:
        import numpy as np
        import scipy.sparse as sp
        import scipy.sparse.linalg as sla
    except Exception as exc:  # broken binary wheels must also fail closed
        raise SparseBackendUnavailable(
            f"SciPy sparse backend could not be imported: {exc}"
        ) from exc
    return np, sp, sla


def _to_scipy_csc(matrix: torch.Tensor):
    np, sp, sla = _scipy_modules()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if matrix.device.type != "cpu":
        raise SparseBackendUnavailable("SciPy sparse backend supports CPU tensors only")
    if matrix.dtype not in (torch.float32, torch.float64):
        raise TypeError("matrix must have float32 or float64 dtype")
    if matrix.layout == torch.sparse_csr:
        matrix = matrix.to_sparse_coo()
    elif matrix.layout != torch.sparse_coo:
        raise TypeError("matrix must be a sparse COO or CSR tensor")
    matrix = matrix.detach().coalesce()
    indices = matrix.indices()
    values = matrix.values()
    if not bool(torch.all(torch.isfinite(values))):
        raise ValueError("matrix contains non-finite values")
    rows = indices[0].numpy()
    columns = indices[1].numpy()
    data = values.numpy()
    return sp.coo_matrix((data, (rows, columns)), shape=matrix.shape).tocsc(), sla


@dataclass(frozen=True)
class SparseFactorization:
    """Reusable optional factorization retaining tensor dtype/device contract."""
    method: str
    shape: tuple[int, int]
    dtype: torch.dtype
    matrix_nnz: int
    factor_nnz: int
    _solve_numpy: Callable

    def solve(self, rhs: torch.Tensor) -> torch.Tensor:
        if rhs.device.type != "cpu":
            raise ValueError("rhs must be a CPU tensor")
        if rhs.dtype != self.dtype:
            raise ValueError("rhs dtype differs from factorization dtype")
        if rhs.ndim not in (1, 2) or rhs.shape[0] != self.shape[0]:
            raise ValueError("rhs has incompatible shape")
        if not bool(torch.all(torch.isfinite(rhs))):
            raise ValueError("rhs contains non-finite values")
        try:
            result = self._solve_numpy(rhs.detach().numpy())
        except Exception as exc:
            raise RuntimeError(f"{self.method} solve failed: {exc}") from exc
        answer = torch.from_numpy(result).to(dtype=self.dtype)
        if not bool(torch.all(torch.isfinite(answer))):
            raise RuntimeError(f"{self.method} solve returned non-finite values")
        return answer

    def __call__(self, rhs: torch.Tensor) -> torch.Tensor:
        return self.solve(rhs)


def factorize_sparse(matrix: torch.Tensor, *, method: str = "splu",
                     drop_tolerance: float = 1e-4,
                     fill_factor: float = 10.0) -> SparseFactorization:
    """Build a SciPy SuperLU or ILU factorization, with no dense fallback.

    ``splu`` is an exact sparse direct solve. ``spilu`` is intended as a
    preconditioner and is not represented as an exact solution method.
    """
    if method not in ("splu", "spilu"):
        raise ValueError("method must be 'splu' or 'spilu'")
    if drop_tolerance < 0 or fill_factor < 1:
        raise ValueError("invalid ILU controls")
    csc, sla = _to_scipy_csc(matrix)
    try:
        if method == "splu":
            factor = sla.splu(csc)
        else:
            factor = sla.spilu(csc, drop_tol=drop_tolerance,
                               fill_factor=fill_factor)
    except Exception as exc:
        raise RuntimeError(f"{method} factorization failed: {exc}") from exc
    factor_nnz = int(factor.L.nnz + factor.U.nnz)
    return SparseFactorization(method, tuple(matrix.shape), matrix.dtype,
                               int(csc.nnz), factor_nnz, factor.solve)
