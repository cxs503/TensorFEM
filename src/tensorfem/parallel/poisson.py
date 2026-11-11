"""Distributed Q1 Poisson solver: ghosted vectors, halo exchange, parallel CG.

This is the reference parallel-finite-element path used to measure MPI-style
strong/weak scaling in TensorFEM.  It implements the standard non-overlapping
(partitioned) FEM parallelisation with *one* halo exchange per matrix-vector
product:

1. elements are partitioned (RCB by default) -> element owner;
2. every node is owned by exactly one rank (owner of its lowest-id element,
   resolved with a single ``Allreduce(MIN)`` over the global node array);
3. each rank keeps an *extended element set* = all elements touching one of its
   owned nodes, so its owned-node results need no reduction afterwards;
4. node values of ghost nodes are refreshed with one symmetric neighbour
   exchange per matvec (owner -> ghost), i.e. the PETSc "ghosted vector"
   pattern;
5. CG inner products use ``Allreduce(SUM)``; the two scalars of an iteration
   (p.Ap and r.z) are fused into one collective.

Matrix-free element-by-element (EBE) operator: the element matrix is the exact
2x2-Gauss Q1 Laplacian, applied as a gather -> (me,4)@(4,4) -> scatter-add.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .comm import Comm, SerialComm
from .partition import (PartitionStats, element_adjacency, graph_partition,
                        partition_stats, rcb_partition)

D = torch.float64

__all__ = ["DistPoisson", "q1_laplace_element", "q1_mass_lumped_element", "build"]


def q1_laplace_element(hx: float, hy: float | None = None) -> np.ndarray:
    """Exact Q1 element Laplacian on an ``hx`` x ``hy`` rectangle (2x2 Gauss).

    For a square (``hx == hy``) this reduces to the textbook
    ``(1/6) [[4,-1,-2,-1],[-1,4,-1,-2],[-2,-1,4,-1],[-1,-2,-1,4]]``.
    """
    hy = hx if hy is None else hy
    gp = np.array([-1.0, 1.0]) / np.sqrt(3.0)
    cx = hy / hx              # (2/hx)^2 * (hx*hy/4)
    cy = hx / hy              # (2/hy)^2 * (hx*hy/4)
    Ke = np.zeros((4, 4))
    for xi in gp:
        for eta in gp:
            dxi = np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)]) / 4.0
            det = np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]) / 4.0
            Ke += cx * np.outer(dxi, dxi) + cy * np.outer(det, det)
    return Ke


def q1_mass_lumped_element(hx: float, hy: float | None = None) -> np.ndarray:
    hy = hx if hy is None else hy
    return np.full(4, hx * hy / 4.0)


def exact_solution(x, y):
    """Manufactured solution: u = x(1-x) y(1-y), zero on all four sides.

    Deliberately *not* a trigonometric eigenvector of the discrete Laplacian:
    a single sine mode makes b an exact eigenvector of A*D^-1, so CG converges
    in one iteration and the timing would measure nothing.  This quartic keeps
    CG work realistic while still having a closed-form reference.
    """
    return x * (1.0 - x) * y * (1.0 - y)


def exact_source(x, y):
    """f = -laplace(u) for the quartic manufactured solution above."""
    return 2.0 * (x - x * x + y - y * y)


class DistPoisson:
    """Distributed Q1 Poisson on the unit square (Dirichlet on all sides)."""

    def __init__(self, comm: Comm | None = None, nx: int = 64, ny: int | None = None,
                 nparts: int | None = None, *, partitioner: str = "rcb"):
        self.comm = comm if comm is not None else SerialComm()
        self.rank = self.comm.rank
        self.size = self.comm.size
        self.nx = int(nx)
        self.ny = int(ny if ny is not None else nx)
        self.partitioner = partitioner
        want = nparts if nparts is not None else max(1, self.size)
        self.nparts = max(1, min(int(want), self.nx * self.ny))
        t0 = time.perf_counter()
        self._build_light()
        self._resolve_owner()
        self.setup_seconds = time.perf_counter() - t0

    # -- geometry -----------------------------------------------------------
    def _build_light(self) -> None:
        nx, ny = self.nx, self.ny
        self.hx = 1.0 / nx
        self.hy = 1.0 / ny
        self.h = self.hx
        self.nglob = (nx + 1) * (ny + 1)
        ne = nx * ny
        self.ne_glob = ne
        ex = np.arange(ne) % nx
        ey = np.arange(ne) // nx
        n0 = ey * (nx + 1) + ex
        self.conn = np.stack([n0, n0 + 1, n0 + (nx + 1) + 1, n0 + (nx + 1)], axis=1)
        cen = np.stack([(ex + 0.5) / nx, (ey + 0.5) / ny], axis=1)
        if self.partitioner == "rcb":
            self.elem_owner = rcb_partition(cen, self.nparts)
        elif self.partitioner == "graph":
            self.elem_owner = graph_partition(element_adjacency(self.conn), self.nparts)
        else:  # pragma: no cover
            raise ValueError(f"unknown partitioner {self.partitioner!r}")
        self.stats: PartitionStats | None = None
        if self.rank == 0:
            self.stats = partition_stats(element_adjacency(self.conn), self.elem_owner)

    def _resolve_owner(self) -> None:
        ne = self.ne_glob
        big = np.iinfo(np.int64).max
        best = np.full(self.nglob, big, dtype=np.int64)
        np.minimum.at(best, self.conn.ravel(), np.repeat(np.arange(ne), 4))
        # MIN over ranks; all ids < 2**53 so float64 carries them exactly
        best = self.comm.allreduce(best.astype(np.float64), op="min").astype(np.int64)
        self.node_owner = self.elem_owner[best]

        r = self.rank
        self.owned = np.flatnonzero(self.node_owner == r)
        self.nown = int(self.owned.size)

        node_owned = self.node_owner == r
        ext_mask = node_owned[self.conn].any(axis=1)
        self.ext_elem = np.flatnonzero(ext_mask)
        self.ghost = np.unique(self.conn[ext_mask][~node_owned[self.conn[ext_mask]]])
        self.nghost = int(self.ghost.size)

        self.loc = np.full(self.nglob, -1, dtype=np.int64)
        self.loc[self.owned] = np.arange(self.nown)
        self.loc[self.ghost] = self.nown + np.arange(self.nghost)
        self.conn_loc = self.loc[self.conn[self.ext_elem]]
        if (self.conn_loc < 0).any():  # pragma: no cover - defensive
            raise RuntimeError("extended element set references a non-local node")

        # need[n, j] == True iff node n belongs to an element that owns (at
        # least one node of) rank j, i.e. n lies in rank j's extended element
        # set and therefore must be visible to j as a ghost.  Because the
        # extended set of j reaches one element *beyond* j's own elements, the
        # criterion is "node of an element touching a j-owned node", not "node
        # of a j-owned element" -- using the latter silently drops corner
        # values (verified by the ghost-exchange unit test).
        need = np.zeros((self.nglob, self.size), dtype=bool)
        own_node = self.node_owner[self.conn]              # (ne, 4) owner per node
        for a in range(4):
            for bidx in range(4):
                need[self.conn[:, a], own_node[:, bidx]] = True

        self._plan: dict[int, dict] = {}
        neigh: set[int] = set()
        for j in range(self.size):
            if j == r:
                continue
            send = self.owned[need[self.owned, j]]
            recv = np.flatnonzero(self.node_owner[self.ghost] == j)
            if send.size or recv.size:
                neigh.add(j)
            self._plan[j] = {
                "send_loc": self.loc[send] if send.size else np.zeros(0, np.int64),
                "recv_loc": self.nown + recv,
                "recv_n": int(recv.size),
            }
        self.neighbors = sorted(neigh)
        self.halo_bytes = int(sum(self._plan[j]["recv_n"] for j in self.neighbors) * 8)

        self.Ke = torch.tensor(q1_laplace_element(self.hx, self.hy), dtype=D)
        self._conn_t = torch.tensor(self.conn_loc, dtype=torch.long)
        self._flat_conn = self._conn_t.ravel()

        Ke_diag = np.diag(q1_laplace_element(self.hx, self.hy))
        diag = np.zeros(self.nglob)
        for a in range(4):
            np.add.at(diag, self.conn[self.ext_elem, a], Ke_diag[a])
        diag[diag == 0] = 1.0
        self.diag = torch.tensor(diag[self.owned], dtype=D)

        nxg = self.nx + 1
        ix = self.owned % nxg
        iy = self.owned // nxg
        self.free = ~((ix == 0) | (iy == 0) | (ix == self.nx) | (iy == self.ny))
        self._free_t = torch.tensor(self.free, dtype=torch.bool)

    # -- vector operations --------------------------------------------------
    def halo(self, u: torch.Tensor) -> torch.Tensor:
        """Refresh ghost entries of a local vector with one neighbour exchange."""
        if not self.neighbors:
            return u
        sends = {j: u[self._plan[j]["send_loc"]].numpy() for j in self.neighbors}
        recv_sizes = {j: self._plan[j]["recv_n"] for j in self.neighbors}
        got = self.comm.exchange(sends, recv_sizes)
        for j, vals in got.items():
            if vals.size:
                u[self._plan[j]["recv_loc"]] = torch.from_numpy(vals)
        return u

    def matvec(self, x_owned: torch.Tensor) -> torch.Tensor:
        u = torch.zeros(self.nown + self.nghost, dtype=D)
        u[:self.nown] = x_owned
        self.halo(u)
        ue = u[self._conn_t]                                # (me, 4)
        ye = ue @ self.Ke.t()                               # (me, 4)
        res = torch.zeros(self.nown + self.nghost, dtype=D)
        res.index_add_(0, self._flat_conn, ye.ravel())
        y = res[:self.nown].clone()
        y[~self._free_t] = 0.0                              # Dirichlet rows
        return y

    def assemble_rhs(self, src=exact_source) -> torch.Tensor:
        """Consistent load vector of the manufactured source ``src(x, y)``.

        2x2 Gauss is exact here: N_a is bilinear (deg 1 per direction) and the
        quartic-source gradient gives deg 2, product deg 3 <= 2*2-1.
        """
        nx, hx, hy = self.nx, self.hx, self.hy
        ex = self.ext_elem % nx
        ey = self.ext_elem // nx
        gp = np.array([-1.0, 1.0]) / np.sqrt(3.0)
        b = np.zeros(self.nown + self.nghost)
        for xi in gp:
            for eta in gp:
                xc = (ex + 0.5 * (1 + xi)) * hx
                yc = (ey + 0.5 * (1 + eta)) * hy
                fval = src(xc, yc)
                N = np.stack([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                              (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)]) / 4.0
                w = (hx / 2.0) * (hy / 2.0)
                for a in range(4):
                    np.add.at(b, self.conn_loc[:, a], w * N[a] * fval)
        out = torch.tensor(b[:self.nown], dtype=D)
        out[~self._free_t] = 0.0
        return out

    # -- solver -------------------------------------------------------------
    def cg(self, b: torch.Tensor, *, iters: int | None = None,
           tol: float = 1e-10, jacobi: bool = True) -> dict:
        """Distributed preconditioned CG; ``iters`` forces a fixed budget."""
        x = torch.zeros_like(b)
        pre = self.diag if jacobi else torch.ones_like(self.diag)
        r = b - self.matvec(x)
        z = r / pre
        p = z.clone()
        r0 = float(self.comm.allreduce(np.array([float((r * r).sum())]), op="sum")[0])
        rz = float(self.comm.allreduce(np.array([float((r * z).sum())]), op="sum")[0])
        rz0 = rz
        it = 0
        nxt = r0
        budget = iters if iters is not None else 20000
        while it < budget:
            it += 1
            Ap = self.matvec(p)
            two = self.comm.allreduce(
                np.array([float((p * Ap).sum()), float((r * z).sum())]), op="sum")
            pAp, rz = float(two[0]), float(two[1])
            if pAp == 0.0:  # pragma: no cover
                break
            alpha = rz / pAp
            x += alpha * p
            r -= alpha * Ap
            z = r / pre
            # single collective per iteration: the preconditioned residual norm
            # serves both the beta recurrence and the stopping test
            rz_new = float(self.comm.allreduce(np.array([float((r * z).sum())]),
                                               op="sum")[0])
            if iters is None and rz_new <= (tol ** 2) * rz0:
                break
            beta = rz_new / rz if rz else 0.0
            p = z + beta * p
            rz = rz_new
        return {
            "iters": it,
            "residual": float(np.sqrt(max(nxt, 0.0))),
            "residual_rel": float(np.sqrt(max(nxt, 0.0) / r0)) if r0 else 0.0,
            "r_norm0": float(np.sqrt(max(r0, 0.0))),
            "x": x,
        }

    def l2_error(self, x_owned: torch.Tensor, uex=exact_solution,
                 order: int = 5) -> float:
        """L2 error against the manufactured solution (n x n Gauss integration).

        ``order=5`` integrates the squared error of a quartic field essentially
        exactly, so the reported norm is the true L2 norm of (u_h - u).
        """
        u = torch.zeros(self.nown + self.nghost, dtype=D)
        u[:self.nown] = x_owned
        self.halo(u)
        nx, hx, hy = self.nx, self.hx, self.hy
        # integrate over OWNED elements only: extended sets overlap between
        # ranks, so integrating them would multiply the norm.
        own = self.elem_owner[self.ext_elem] == self.rank
        ex = self.ext_elem[own] % nx
        ey = self.ext_elem[own] // nx
        gp, gw = np.polynomial.legendre.leggauss(order)
        wgt = (hx / 2.0) * (hy / 2.0)
        ue = u[self._conn_t[own]].numpy()
        err = 0.0
        for a_, xi in enumerate(gp):
            for b_, eta in enumerate(gp):
                xc = (ex + 0.5 * (1 + xi)) * hx
                yc = (ey + 0.5 * (1 + eta)) * hy
                exact = uex(xc, yc)
                N = np.stack([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                              (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)]) / 4.0
                uh = ue @ N
                err += float(((uh - exact) ** 2).sum()) * wgt * gw[a_] * gw[b_]
        return float(np.sqrt(self.comm.allreduce(np.array([err]), op="sum")[0]))

    def global_dofs(self) -> int:
        return int(self.comm.allreduce(np.array([float(self.nown)]), op="sum")[0])


def build(comm: Comm, nx: int, ny: int | None = None, nparts: int | None = None,
          *, partitioner: str = "rcb") -> DistPoisson:
    """Convenience constructor mirroring ``DistPoisson(comm, ...)``."""
    return DistPoisson(comm, nx, ny, nparts, partitioner=partitioner)