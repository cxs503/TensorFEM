"""Sparse assembly, constraints and iterative linear statics.

The routines deliberately use only public PyTorch operations and preserve
autograd through element values.  Indices/connectivity are discrete inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class MPC:
    """Linear constraint ``u[slave] = offset + sum(c*u[master])``."""

    slave: int
    masters: tuple[tuple[int, float], ...]
    offset: float = 0.0


class DofManager:
    """Deterministic node/component to global/equation numbering."""

    def __init__(self, nnode: int, components: Sequence[str]):
        if nnode <= 0 or not components:
            raise ValueError("nnode and components must be non-empty")
        self.nnode = int(nnode)
        self.components = tuple(components)
        if len(set(self.components)) != len(self.components):
            raise ValueError("component names must be unique")
        self.ndof_per_node = len(self.components)
        self.ndof = self.nnode * self.ndof_per_node

    def dof(self, node: int, component: str | int) -> int:
        if not 0 <= node < self.nnode:
            raise IndexError("node out of range")
        c = self.components.index(component) if isinstance(component, str) else component
        if not 0 <= c < self.ndof_per_node:
            raise IndexError("component out of range")
        return node * self.ndof_per_node + c

    def element_dofs(self, nodes: Tensor) -> Tensor:
        nodes = torch.as_tensor(nodes, dtype=torch.long)
        c = torch.arange(self.ndof_per_node, device=nodes.device)
        return (nodes[..., None] * self.ndof_per_node + c).reshape(*nodes.shape[:-1], -1)

    def equations(self, fixed: Iterable[int]) -> Tensor:
        fixed_t = torch.as_tensor(tuple(fixed), dtype=torch.long)
        if fixed_t.numel() and (fixed_t.min() < 0 or fixed_t.max() >= self.ndof):
            raise IndexError("fixed DOF out of range")
        eq = torch.full((self.ndof,), -1, dtype=torch.long)
        mask = torch.ones(self.ndof, dtype=torch.bool)
        mask[fixed_t] = False
        eq[mask] = torch.arange(int(mask.sum()))
        return eq


def assemble_coo(element_matrices: Tensor, element_dofs: Tensor, ndof: int,
                 *, csr: bool = False) -> Tensor:
    """Vectorized assembly of equal-sized element matrices."""
    if element_matrices.ndim != 3 or element_matrices.shape[1] != element_matrices.shape[2]:
        raise ValueError("element_matrices must have shape (ne,nen,nen)")
    if element_dofs.shape != element_matrices.shape[:2]:
        raise ValueError("element_dofs must have shape (ne,nen)")
    if element_dofs.numel() and (element_dofs.min() < 0 or element_dofs.max() >= ndof):
        raise IndexError("element DOF out of range")
    n = element_dofs.shape[1]
    rows = element_dofs[:, :, None].expand(-1, n, n).reshape(-1)
    cols = element_dofs[:, None, :].expand(-1, n, n).reshape(-1)
    # Rows/columns were range-checked above. PyTorch 2.12 requires its policy
    # context (the constructor keyword alone still emits an implicit-policy
    # warning); avoid a second O(nnz) validation pass here.
    with torch.sparse.check_sparse_tensor_invariants(enable=False):
        K = torch.sparse_coo_tensor(
            torch.stack((rows, cols)), element_matrices.reshape(-1),
            (ndof, ndof), device=element_matrices.device,
            check_invariants=False).coalesce()
    return K.to_sparse_csr() if csr else K


def sparse_mv(A: Tensor, x: Tensor) -> Tensor:
    if A.layout not in (torch.sparse_coo, torch.sparse_csr):
        return A @ x
    return torch.sparse.mm(A, x[:, None])[:, 0]


@dataclass(frozen=True)
class IterativeResult:
    x: Tensor
    converged: bool
    iterations: int
    relative_residual: float


def conjugate_gradient(A: Tensor | Callable[[Tensor], Tensor], b: Tensor, *,
                       x0: Tensor | None = None, rtol: float = 1e-10,
                       atol: float = 0.0, maxiter: int | None = None,
                       diagonal: Tensor | None = None) -> IterativeResult:
    """Preconditioned CG for symmetric positive-definite operators."""
    if b.ndim != 1:
        raise ValueError("b must be one-dimensional")
    mv = A if callable(A) else lambda x: sparse_mv(A, x)
    n = b.numel(); maxiter = maxiter or max(10, 2 * n)
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - mv(x)
    norm_b = float(torch.linalg.vector_norm(b))
    threshold = max(atol, rtol * norm_b)
    if float(torch.linalg.vector_norm(r)) <= threshold:
        return IterativeResult(x, True, 0, 0.0)
    if diagonal is not None:
        if torch.any(diagonal <= 0):
            raise ValueError("Jacobi diagonal must be positive")
        pre = lambda v: v / diagonal
    else:
        pre = lambda v: v
    z = pre(r); p = z.clone(); rz = torch.dot(r, z)
    for it in range(1, maxiter + 1):
        Ap = mv(p); curvature = torch.dot(p, Ap)
        if float(curvature) <= 0:
            raise RuntimeError("CG operator is not positive definite")
        alpha = rz / curvature; x = x + alpha * p; r = r - alpha * Ap
        nr = float(torch.linalg.vector_norm(r))
        if nr <= threshold:
            return IterativeResult(x, True, it, nr / max(norm_b, 1e-300))
        z = pre(r); rz_new = torch.dot(r, z)
        p = z + (rz_new / rz) * p; rz = rz_new
    nr = float(torch.linalg.vector_norm(r))
    return IterativeResult(x, False, maxiter, nr / max(norm_b, 1e-300))


def constraint_transform(ndof: int, dirichlet: Mapping[int, float] | None = None,
                         mpcs: Sequence[MPC] = (), *, dtype=torch.float64,
                         device=None) -> tuple[Tensor, Tensor, tuple[int, ...]]:
    """Return sparse ``T`` and ``u0`` such that ``u=T q+u0``.

    MPC masters must be independent DOFs. Chained/cyclic MPCs are rejected,
    making the elimination deterministic and fail-closed.
    """
    prescribed = dict(dirichlet or {})
    slave_map = {m.slave: m for m in mpcs}
    if len(slave_map) != len(mpcs):
        raise ValueError("an MPC slave may occur only once")
    if set(prescribed) & set(slave_map):
        raise ValueError("a DOF cannot be both prescribed and an MPC slave")
    constrained = set(prescribed) | set(slave_map)
    if any(i < 0 or i >= ndof for i in constrained):
        raise IndexError("constraint DOF out of range")
    independent = tuple(i for i in range(ndof) if i not in constrained)
    column = {d: j for j, d in enumerate(independent)}
    rows: list[int] = []; cols: list[int] = []; vals: list[float] = []
    u0 = torch.zeros(ndof, dtype=dtype, device=device)
    for d, v in prescribed.items(): u0[d] = v
    for d in independent:
        rows.append(d); cols.append(column[d]); vals.append(1.0)
    for m in mpcs:
        u0[m.slave] = m.offset
        for master, coefficient in m.masters:
            if master < 0 or master >= ndof:
                raise IndexError("MPC master DOF out of range")
            if master not in column:
                raise ValueError("MPC masters must be unconstrained independent DOFs")
            rows.append(m.slave); cols.append(column[master]); vals.append(coefficient)
    idx = torch.tensor((rows, cols), dtype=torch.long, device=device)
    val = torch.tensor(vals, dtype=dtype, device=device)
    # Indices are generated locally from validated DOFs, so the tensor is safe
    # by construction and does not need PyTorch's duplicate O(nnz) scan.
    with torch.sparse.check_sparse_tensor_invariants(enable=False):
        T = torch.sparse_coo_tensor(idx, val, (ndof, len(independent)),
                                    check_invariants=False).coalesce()
    return T, u0, independent


@dataclass(frozen=True)
class SparseStaticResult:
    displacement: Tensor
    reaction: Tensor
    iterative: IterativeResult


def solve_sparse_static(K: Tensor, force: Tensor, *,
                        dirichlet: Mapping[int, float] | None = None,
                        mpcs: Sequence[MPC] = (), rtol: float = 1e-10,
                        maxiter: int | None = None) -> SparseStaticResult:
    """Solve constrained SPD statics without forming a dense reduced matrix."""
    if K.shape != (force.numel(), force.numel()):
        raise ValueError("K and force sizes differ")
    if K.layout not in (torch.sparse_coo, torch.sparse_csr):
        K = K.to_sparse_coo().coalesce()
    T, u0, _ = constraint_transform(force.numel(), dirichlet, mpcs,
                                     dtype=force.dtype, device=force.device)
    TT = T.transpose(0, 1)
    def reduced_mv(q: Tensor) -> Tensor:
        return sparse_mv(TT, sparse_mv(K, sparse_mv(T, q)))
    rhs = sparse_mv(TT, force - sparse_mv(K, u0))
    solved = conjugate_gradient(reduced_mv, rhs, rtol=rtol, maxiter=maxiter)
    if not solved.converged:
        raise RuntimeError(f"CG did not converge: residual={solved.relative_residual:g}")
    u = sparse_mv(T, solved.x) + u0
    return SparseStaticResult(u, sparse_mv(K, u) - force, solved)
