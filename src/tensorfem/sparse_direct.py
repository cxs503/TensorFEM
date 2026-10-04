"""Optional CPU sparse direct and ILU adapters.

SciPy is deliberately imported only when a factorization is requested.  The
core TensorFEM installation therefore has no NumPy/SciPy dependency.  Missing
or unsupported backends fail closed instead of silently densifying a matrix or
changing the requested algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from time import perf_counter
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


def sparse_backend_statuses() -> tuple[SparseBackendStatus, ...]:
    """Probe supported industrial sparse backends without importing them.

    Availability means that the Python binding required by TensorFEM exists;
    finding a system shared library alone is deliberately not sufficient.
    """
    scipy = scipy_sparse_status()
    optional = (
        ("petsc", "petsc4py", "petsc4py is not installed"),
        ("pardiso", "pypardiso", "pypardiso is not installed"),
        ("suitesparse", "sksparse", "scikit-sparse is not installed"),
        ("amg", "pyamg", "pyamg is not installed"),
    )
    rows = [SparseBackendStatus(
        "superlu", scipy.available,
        "SciPy SuperLU available" if scipy.available else scipy.reason,
    )]
    for name, module, absent in optional:
        rows.append(SparseBackendStatus(
            name, find_spec(module) is not None,
            "available" if find_spec(module) is not None else absent,
        ))
    # MUMPS has several incompatible bindings.  Only a supported binding may
    # be selected; a bare libdmumps cannot satisfy the Python tensor contract.
    mumps_modules = ("mumps", "pymumps", "mumpspy")
    found = next((module for module in mumps_modules if find_spec(module)), None)
    rows.append(SparseBackendStatus(
        "mumps", found is not None,
        f"{found} available" if found else "no supported MUMPS Python binding is installed",
    ))
    return tuple(rows)


@dataclass(frozen=True)
class SparseSolveDiagnostics:
    backend: str
    shape: tuple[int, int]
    matrix_nnz: int
    factor_nnz: int
    factorization_seconds: float
    solve_seconds: float
    solve_calls: int
    right_hand_sides: int
    estimated_matrix_bytes: int
    estimated_factor_bytes: int
    last_relative_residual: float | None


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


@dataclass
class SparseFactorization:
    """Reusable optional factorization retaining tensor dtype/device contract."""
    method: str
    shape: tuple[int, int]
    dtype: torch.dtype
    matrix_nnz: int
    factor_nnz: int
    _solve_numpy: Callable
    factorization_seconds: float = 0.0
    solve_seconds: float = 0.0
    solve_calls: int = 0
    right_hand_sides: int = 0
    last_relative_residual: float | None = None
    _matrix_action: Callable | None = None

    def solve(self, rhs: torch.Tensor) -> torch.Tensor:
        if rhs.device.type != "cpu":
            raise ValueError("rhs must be a CPU tensor")
        if rhs.dtype != self.dtype:
            raise ValueError("rhs dtype differs from factorization dtype")
        if rhs.ndim not in (1, 2) or rhs.shape[0] != self.shape[0]:
            raise ValueError("rhs has incompatible shape")
        if not bool(torch.all(torch.isfinite(rhs))):
            raise ValueError("rhs contains non-finite values")
        started = perf_counter()
        try:
            result = self._solve_numpy(rhs.detach().numpy())
        except Exception as exc:
            raise RuntimeError(f"{self.method} solve failed: {exc}") from exc
        answer = torch.from_numpy(result).to(dtype=self.dtype)
        if not bool(torch.all(torch.isfinite(answer))):
            raise RuntimeError(f"{self.method} solve returned non-finite values")
        self.solve_seconds += perf_counter() - started
        self.solve_calls += 1
        self.right_hand_sides += 1 if rhs.ndim == 1 else rhs.shape[1]
        if self._matrix_action is not None:
            residual = self._matrix_action(answer) - rhs
            scale = max(float(torch.linalg.vector_norm(rhs)), 1.0)
            self.last_relative_residual = float(torch.linalg.vector_norm(residual)) / scale
        return answer

    def __call__(self, rhs: torch.Tensor) -> torch.Tensor:
        return self.solve(rhs)

    @property
    def diagnostics(self) -> SparseSolveDiagnostics:
        value_bytes = torch.empty((), dtype=self.dtype).element_size()
        # CSC uses one integer per row index and n+1 column pointers.  SuperLU
        # exposes only aggregate L/U nnz, so factor storage is an estimate.
        index_bytes = 4
        n = self.shape[0]
        return SparseSolveDiagnostics(
            self.method, self.shape, self.matrix_nnz, self.factor_nnz,
            self.factorization_seconds, self.solve_seconds, self.solve_calls,
            self.right_hand_sides,
            self.matrix_nnz*(value_bytes+index_bytes)+(n+1)*index_bytes,
            self.factor_nnz*(value_bytes+index_bytes)+2*(n+1)*index_bytes,
            self.last_relative_residual,
        )


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
    started = perf_counter()
    try:
        if method == "splu":
            factor = sla.splu(csc)
        else:
            factor = sla.spilu(csc, drop_tol=drop_tolerance,
                               fill_factor=fill_factor)
    except Exception as exc:
        raise RuntimeError(f"{method} factorization failed: {exc}") from exc
    factor_nnz = int(factor.L.nnz + factor.U.nnz)
    elapsed = perf_counter() - started
    coalesced = matrix.coalesce() if matrix.layout == torch.sparse_coo else matrix.to_sparse_coo().coalesce()
    action = lambda value: torch.sparse.mm(
        coalesced, value[:, None] if value.ndim == 1 else value
    ).squeeze(1) if value.ndim == 1 else torch.sparse.mm(coalesced, value)
    return SparseFactorization(method, tuple(matrix.shape), matrix.dtype,
                               int(csc.nnz), factor_nnz, factor.solve,
                               factorization_seconds=elapsed,
                               _matrix_action=action)


class SparseLinearSolver:
    """Explicit, reusable sparse linear solver adapter.

    No dense or alternate-backend fallback is performed. ``auto`` selects a
    backend only when exactly the documented priority candidate is available;
    in this release that candidate is SciPy SuperLU.
    """
    def __init__(self, backend: str = "auto") -> None:
        if backend not in ("auto", "superlu"):
            known = {row.name: row for row in sparse_backend_statuses()}
            if backend in known and not known[backend].available:
                raise SparseBackendUnavailable(known[backend].reason)
            raise SparseBackendUnavailable(f"backend '{backend}' is not implemented")
        status = next(row for row in sparse_backend_statuses() if row.name == "superlu")
        if not status.available:
            raise SparseBackendUnavailable(status.reason)
        self.backend = "superlu"

    def factorize(self, matrix: torch.Tensor) -> SparseFactorization:
        return factorize_sparse(matrix, method="splu")
