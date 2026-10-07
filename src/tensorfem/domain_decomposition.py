"""Deterministic domain decomposition for sparse symmetric systems.

The implementation is deliberately single-process first: the same subdomain
maps can be assigned to ranks, while tests and reproducibility need no process
launcher.  Overlapping additive Schwarz is used as an SPD preconditioner.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from .sparse_core import IterativeResult, sparse_mv

Tensor = torch.Tensor


@dataclass(frozen=True)
class Subdomain:
    id: int
    owned: Tensor
    local: Tensor
    interface: Tensor

    def __post_init__(self) -> None:
        for name in ("owned", "local", "interface"):
            value = getattr(self, name)
            if value.dtype != torch.long or value.ndim != 1:
                raise TypeError(f"{name} must be a one-dimensional long tensor")

    @property
    def local_to_global(self) -> Tensor:
        return self.local

    def global_to_local(self, ndof: int) -> Tensor:
        out = torch.full((ndof,), -1, dtype=torch.long, device=self.local.device)
        out[self.local] = torch.arange(self.local.numel(), device=self.local.device)
        return out


def contiguous_owners(ndof: int, nparts: int, *, device=None) -> Tensor:
    """Balanced deterministic ownership for consecutively numbered DOFs."""
    if ndof < 1 or nparts < 1 or nparts > ndof:
        raise ValueError("require 1 <= nparts <= ndof")
    return torch.div(torch.arange(ndof, device=device) * nparts, ndof,
                     rounding_mode="floor")


def graph_owners(A: Tensor, nparts: int) -> Tensor:
    """Deterministic graph-aware greedy partitioning without dependencies.

    Vertices are visited by descending degree (DOF breaks ties).  Assignment
    prefers the partition containing most assigned neighbours, subject to the
    exact balanced capacity.  This is a portable baseline, not a METIS claim.
    """
    rows, cols = _coo_indices(A)
    n = A.shape[0]
    if nparts < 1 or nparts > n:
        raise ValueError("require 1 <= nparts <= ndof")
    neighbours = [set() for _ in range(n)]
    for i, j in zip(rows.tolist(), cols.tolist()):
        if i != j:
            neighbours[i].add(j); neighbours[j].add(i)
    order = sorted(range(n), key=lambda i: (-len(neighbours[i]), i))
    cap = [(n + p) // nparts for p in range(nparts)]
    # Correct capacities to sum exactly and differ by at most one.
    base, rem = divmod(n, nparts); cap = [base + (p < rem) for p in range(nparts)]
    owners = [-1] * n; used = [0] * nparts
    for i in order:
        score = [sum(owners[j] == p for j in neighbours[i]) if used[p] < cap[p] else -1
                 for p in range(nparts)]
        best = max(range(nparts), key=lambda p: (score[p], -used[p], -p))
        owners[i] = best; used[best] += 1
    return torch.tensor(owners, dtype=torch.long, device=A.device)


def build_subdomains(A: Tensor, owners: Tensor, *, overlap: int = 1) -> tuple[Subdomain, ...]:
    """Build owned, halo and interface maps, rejecting invalid partitions."""
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    n = A.shape[0]
    if owners.dtype != torch.long or owners.shape != (n,):
        raise TypeError("owners must be a long tensor of length ndof")
    if overlap < 0:
        raise ValueError("overlap must be nonnegative")
    labels = torch.unique(owners, sorted=True)
    if labels.numel() == 0 or not torch.equal(labels.cpu(), torch.arange(labels.numel())):
        raise ValueError("partition labels must be contiguous from zero and nonempty")
    rows, cols = _coo_indices(A)
    adjacency = [set() for _ in range(n)]
    for i, j in zip(rows.tolist(), cols.tolist()):
        if i != j:
            adjacency[i].add(j); adjacency[j].add(i)
    result = []
    for p in labels.tolist():
        own = set(torch.nonzero(owners == p, as_tuple=False)[:, 0].tolist())
        interface = {i for i in own if any(int(owners[j]) != p for j in adjacency[i])}
        local = set(own); frontier = set(own)
        for _ in range(overlap):
            frontier = set().union(*(adjacency[i] for i in frontier)) - local
            local |= frontier
        make = lambda s: torch.tensor(sorted(s), dtype=torch.long, device=A.device)
        result.append(Subdomain(p, make(own), make(local), make(interface)))
    if sum(s.owned.numel() for s in result) != n:
        raise RuntimeError("partition does not cover every DOF exactly once")
    return tuple(result)


def _coo_indices(A: Tensor) -> tuple[Tensor, Tensor]:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    C = A.to_sparse_coo().coalesce() if A.layout != torch.sparse_coo else A.coalesce()
    return C.indices()[0], C.indices()[1]


class AdditiveSchwarz:
    """Exact local overlapping additive-Schwarz inverse operator."""
    def __init__(self, A: Tensor, subdomains: Sequence[Subdomain]):
        if not subdomains:
            raise ValueError("at least one subdomain is required")
        self.A = A; self.subdomains = tuple(subdomains); self.n = A.shape[0]
        dense = A.to_dense() if A.layout != torch.strided else A
        multiplicity = torch.zeros(self.n, dtype=dense.dtype, device=dense.device)
        for s in self.subdomains: multiplicity[s.local] += 1
        if bool(torch.any(multiplicity == 0)):
            raise ValueError("subdomains do not cover every DOF")
        self.weights = multiplicity.rsqrt()
        self.local_factors = []
        for s in self.subdomains:
            local = dense[s.local][:, s.local]
            factor, info = torch.linalg.cholesky_ex(local)
            if int(info) != 0:
                raise ValueError(f"subdomain {s.id} matrix is not positive definite")
            self.local_factors.append(factor)

    def __call__(self, residual: Tensor) -> Tensor:
        if residual.shape != (self.n,):
            raise ValueError("residual size differs from decomposition")
        out = torch.zeros_like(residual)
        # Symmetric partition-of-unity scaling preserves PCG requirements.
        for s, L in zip(self.subdomains, self.local_factors):
            rhs = residual[s.local] * self.weights[s.local]
            z = torch.cholesky_solve(rhs[:, None], L)[:, 0]
            out.index_add_(0, s.local, z * self.weights[s.local])
        return out


def solve_schwarz(A: Tensor, b: Tensor, *, nparts: int = 2, overlap: int = 1,
                  partition: str = "contiguous", rtol: float = 1e-10,
                  maxiter: int | None = None) -> IterativeResult:
    """Solve an SPD system with preconditioned CG and Schwarz local solves."""
    if b.ndim != 1 or A.shape != (b.numel(), b.numel()):
        raise ValueError("incompatible A and b")
    owners = contiguous_owners(b.numel(), nparts, device=b.device) if partition == "contiguous" \
        else graph_owners(A, nparts) if partition == "graph" else None
    if owners is None:
        raise ValueError("partition must be 'contiguous' or 'graph'")
    domains = build_subdomains(A, owners, overlap=overlap)
    pre = AdditiveSchwarz(A, domains)
    return _pcg(lambda x: sparse_mv(A, x), b, pre, rtol=rtol, maxiter=maxiter)


def _pcg(mv: Callable[[Tensor], Tensor], b: Tensor, pre: Callable[[Tensor], Tensor],
         *, rtol: float, maxiter: int | None) -> IterativeResult:
    n = b.numel(); maxiter = maxiter or max(10, 2*n); x = torch.zeros_like(b)
    r = b.clone(); normb = float(torch.linalg.vector_norm(b)); threshold = rtol*normb
    if normb == 0: return IterativeResult(x, True, 0, 0.0)
    z = pre(r); p = z.clone(); rz = torch.dot(r, z)
    if float(rz) <= 0: raise RuntimeError("Schwarz preconditioner is not positive definite")
    for it in range(1, maxiter + 1):
        Ap = mv(p); curvature = torch.dot(p, Ap)
        if float(curvature) <= 0: raise RuntimeError("operator is not positive definite")
        alpha = rz/curvature; x += alpha*p; r -= alpha*Ap
        nr = float(torch.linalg.vector_norm(r))
        if nr <= threshold: return IterativeResult(x, True, it, nr/normb)
        z = pre(r); new = torch.dot(r, z)
        if float(new) <= 0: raise RuntimeError("Schwarz preconditioner broke positivity")
        p = z + new/rz*p; rz = new
    nr = float(torch.linalg.vector_norm(r))
    return IterativeResult(x, False, maxiter, nr/normb)


def distributed_sum(local: Tensor, *, group=None) -> Tensor:
    """Optional rank reduction, failing closed unless distributed is initialized."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed process group is not initialized")
    out = local.clone()
    torch.distributed.all_reduce(out, group=group)
    return out
