"""Cell-level graph-coloring domain decomposition with shared-boundary assembly.

Extends the DOF-level Schwarz machinery to FEM meshes: elements are partitioned
via greedy graph coloring on the element-adjacency graph, then each subdomain
assembles its own local stiffness/load.  Shared (interface) DOFs are identified
and a partition-of-unity weighted additive-Schwarz preconditioner is built.

This module provides both single-process and multi-thread parallel execution.
Thread-level parallelism uses ``concurrent.futures.ThreadPoolExecutor`` for
concurrent subdomain solves without process-spawn overhead.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Sequence

import torch

from .domain_decomposition import (
    AdditiveSchwarz, Subdomain, build_subdomains, contiguous_owners,
    graph_owners, solve_schwarz,
)
from .sparse_core import IterativeResult, sparse_mv

Tensor = torch.Tensor
D = torch.float64


@dataclass(frozen=True)
class CellPartition:
    """Result of cell-level graph-coloring partition."""
    element_owners: Tensor
    dof_owners: Tensor
    subdomain_elements: list[Tensor]
    subdomains: tuple[Subdomain, ...]
    interface_dofs: list[Tensor]


def partition_cells_by_coloring(
    ne: int,
    element_adjacency: list[set[int]],
    ndof: int,
    element_dofs: Tensor,
    nparts: int,
) -> CellPartition:
    """Partition elements via greedy graph coloring, then derive DOF ownership."""
    if nparts < 1 or nparts > ne:
        raise ValueError(f"require 1 <= nparts <= ne (got {nparts}, ne={ne})")

    order = sorted(range(ne), key=lambda e: (-len(element_adjacency[e]), e))
    cap = [ne // nparts + (p < ne % nparts) for p in range(nparts)]
    elem_owner = [-1] * ne
    used = [0] * nparts

    for e in order:
        score = [0] * nparts
        for nb in element_adjacency[e]:
            if elem_owner[nb] >= 0:
                score[elem_owner[nb]] += 1
        best = -1
        best_score = -1
        for p in range(nparts):
            if used[p] < cap[p] and score[p] > best_score:
                best = p
                best_score = score[p]
        if best < 0:
            for p in range(nparts):
                if used[p] < cap[p]:
                    best = p
                    break
        if best < 0:
            best = 0
        elem_owner[e] = best
        used[best] += 1

    elem_owner_t = torch.tensor(elem_owner, dtype=torch.long)

    dof_owner_count = torch.zeros((ndof, nparts), dtype=torch.long)
    for e in range(ne):
        for dof_idx in element_dofs[e].tolist():
            dof_owner_count[dof_idx, elem_owner[e]] += 1
    dof_owners = dof_owner_count.argmax(dim=1)

    sub_elem = []
    for p in range(nparts):
        sub_elem.append(torch.nonzero(elem_owner_t == p, as_tuple=False).squeeze(-1))

    subdomains = build_subdomains_from_dof_owners(ndof, dof_owners, element_dofs, elem_owner_t)

    interface_dofs = []
    for s in subdomains:
        interface_dofs.append(s.interface.clone())

    return CellPartition(
        element_owners=elem_owner_t,
        dof_owners=dof_owners,
        subdomain_elements=sub_elem,
        subdomains=subdomains,
        interface_dofs=interface_dofs,
    )


def build_subdomains_from_dof_owners(
    ndof: int,
    dof_owners: Tensor,
    element_dofs: Tensor,
    elem_owners: Tensor,
    *,
    overlap: int = 1,
) -> tuple[Subdomain, ...]:
    """Build Subdomain objects from DOF ownership, using element adjacency for halo."""
    ne = element_dofs.shape[0]

    dof_adj = [set() for _ in range(ndof)]
    for e in range(ne):
        dofs = element_dofs[e].tolist()
        for i in range(len(dofs)):
            for j in range(i + 1, len(dofs)):
                dof_adj[dofs[i]].add(dofs[j])
                dof_adj[dofs[j]].add(dofs[i])

    rows, cols, vals = [], [], []
    for i in range(ndof):
        rows.append(i); cols.append(i); vals.append(1.0)
        for j in dof_adj[i]:
            rows.append(i); cols.append(j); vals.append(1.0)
    r = torch.tensor(rows, dtype=torch.long)
    c = torch.tensor(cols, dtype=torch.long)
    v = torch.tensor(vals, dtype=D)
    with torch.sparse.check_sparse_tensor_invariants(enable=False):
        A_adj = torch.sparse_coo_tensor(
            torch.stack([r, c]), v, (ndof, ndof),
            check_invariants=False).coalesce()

    return build_subdomains(A_adj, dof_owners, overlap=overlap)


def assemble_subdomain_system(
    K_global: Tensor,
    F_global: Tensor,
    subdomain: Subdomain,
) -> tuple[Tensor, Tensor]:
    """Extract subdomain-local stiffness and load from global assembled system."""
    local = subdomain.local
    K_local = K_global[local][:, local].to_dense()
    F_local = F_global[local].clone()
    return K_local, F_local


def solve_domain_decomposed(
    K_global: Tensor,
    F_global: Tensor,
    nparts: int = 2,
    partition: str = "contiguous",
    overlap: int = 1,
    rtol: float = 1e-11,
    maxiter: int | None = None,
) -> IterativeResult:
    """Solve the global system via domain-decomposed additive Schwarz."""
    return solve_schwarz(
        K_global, F_global, nparts=nparts, overlap=overlap,
        partition=partition, rtol=rtol, maxiter=maxiter,
    )


def solve_global_direct(
    K_global: Tensor,
    F_global: Tensor,
) -> Tensor:
    """Solve the global system directly (dense solve) for reference."""
    K_dense = K_global.to_dense()
    return torch.linalg.solve(K_dense, F_global)


# ---------------------------------------------------------------------------
# Parallel Schwarz: thread-level speedup
# ---------------------------------------------------------------------------

class ParallelAdditiveSchwarz:
    """Additive Schwarz preconditioner with parallel local solves via threads."""
    def __init__(self, K: Tensor, subdomains: Sequence[Subdomain], *, n_workers: int = 2):
        if not subdomains:
            raise ValueError("at least one subdomain is required")
        self.K = K
        self.subdomains = tuple(subdomains)
        self.n = K.shape[0]
        self.n_workers = n_workers

        dense = K.to_dense() if K.layout != torch.strided else K
        multiplicity = torch.zeros(self.n, dtype=dense.dtype)
        for s in self.subdomains:
            multiplicity[s.local] += 1
        if bool(torch.any(multiplicity == 0)):
            raise ValueError("subdomains do not cover every DOF")
        self.weights = multiplicity.rsqrt()

        self.local_matrices = []
        for s in self.subdomains:
            local = dense[s.local][:, s.local]
            self.local_matrices.append(local)

    def __call__(self, residual: Tensor) -> Tensor:
        if residual.shape != (self.n,):
            raise ValueError("residual size differs from decomposition")

        if self.n_workers <= 1:
            return self._sequential_apply(residual)

        # Prepare tasks
        tasks = []
        for s, K_local in zip(self.subdomains, self.local_matrices):
            rhs = residual[s.local] * self.weights[s.local]
            tasks.append((K_local, rhs, self.weights[s.local], s.local))

        # Parallel execution via threads
        out = torch.zeros_like(residual)
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(executor.map(self._solve_one, tasks))
        for contrib, local_idx in zip(results, [t[3] for t in tasks]):
            out.index_add_(0, local_idx, contrib)
        return out

    @staticmethod
    def _solve_one(args: tuple) -> Tensor:
        K_local, rhs, weights, local_idx = args
        L = torch.linalg.cholesky(K_local)
        z = torch.cholesky_solve(rhs[:, None], L)[:, 0]
        return z * weights

    def _sequential_apply(self, residual: Tensor) -> Tensor:
        out = torch.zeros_like(residual)
        for s, K_local in zip(self.subdomains, self.local_matrices):
            rhs = residual[s.local] * self.weights[s.local]
            L = torch.linalg.cholesky(K_local)
            z = torch.cholesky_solve(rhs[:, None], L)[:, 0]
            out.index_add_(0, s.local, z * self.weights[s.local])
        return out


def _pcg_with_pre(mv, b: Tensor, pre, *, rtol: float, maxiter: int | None) -> IterativeResult:
    """PCG with arbitrary preconditioner."""
    n = b.numel(); maxiter = maxiter or max(10, 2*n); x = torch.zeros_like(b)
    r = b.clone(); normb = float(torch.linalg.vector_norm(b)); threshold = rtol*normb
    if normb == 0: return IterativeResult(x, True, 0, 0.0)
    z = pre(r); p = z.clone(); rz = torch.dot(r, z)
    if float(rz) <= 0: raise RuntimeError("preconditioner is not positive definite")
    for it in range(1, maxiter + 1):
        Ap = mv(p); curvature = torch.dot(p, Ap)
        if float(curvature) <= 0: raise RuntimeError("operator is not positive definite")
        alpha = rz/curvature; x += alpha*p; r -= alpha*Ap
        nr = float(torch.linalg.vector_norm(r))
        if nr <= threshold: return IterativeResult(x, True, it, nr/normb)
        z = pre(r); new = torch.dot(r, z)
        if float(new) <= 0: raise RuntimeError("preconditioner broke positivity")
        p = z + new/rz*p; rz = new
    nr = float(torch.linalg.vector_norm(r))
    return IterativeResult(x, False, maxiter, nr/normb)


def solve_schwarz_parallel(
    K: Tensor,
    b: Tensor,
    *,
    nparts: int = 2,
    overlap: int = 1,
    partition: str = "contiguous",
    n_workers: int = 2,
    rtol: float = 1e-10,
    maxiter: int | None = None,
) -> IterativeResult:
    """Solve SPD system with parallel Schwarz-preconditioned CG."""
    if b.ndim != 1 or K.shape != (b.numel(), b.numel()):
        raise ValueError("incompatible K and b")
    owners = contiguous_owners(b.numel(), nparts, device=b.device) if partition == "contiguous" \
        else graph_owners(K, nparts) if partition == "graph" else None
    if owners is None:
        raise ValueError("partition must be 'contiguous' or 'graph'")
    domains = build_subdomains(K, owners, overlap=overlap)
    pre = ParallelAdditiveSchwarz(K, domains, n_workers=n_workers)
    return _pcg_with_pre(lambda x: sparse_mv(K, x), b, pre, rtol=rtol, maxiter=maxiter)


def measure_speedup_parallel(
    K_global: Tensor,
    F_global: Tensor,
    nparts: int = 2,
    n_runs: int = 3,
    n_workers: int = 2,
) -> dict:
    """Measure speedup of parallel domain-decomposed solve vs direct solve."""
    # Warm-up
    solve_global_direct(K_global, F_global)
    solve_schwarz_parallel(K_global, F_global, nparts=nparts, n_workers=n_workers)

    # Time direct solve
    t0 = time.perf_counter()
    for _ in range(n_runs):
        solve_global_direct(K_global, F_global)
    t_direct = (time.perf_counter() - t0) / n_runs

    # Time parallel DD solve
    t0 = time.perf_counter()
    for _ in range(n_runs):
        solve_schwarz_parallel(K_global, F_global, nparts=nparts, n_workers=n_workers)
    t_dd = (time.perf_counter() - t0) / n_runs

    speedup = t_direct / t_dd if t_dd > 0 else float("inf")

    return {
        "direct_time_s": t_direct,
        "dd_time_s": t_dd,
        "speedup": speedup,
        "n_workers": n_workers,
        "nparts": nparts,
    }
