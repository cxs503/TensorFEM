"""Mesh partitioning for domain decomposition.

Two standard, dependency-free partitioners plus the metrics used to judge a
partition before any solve is run (a partition is only accepted if it is
balanced and the edge cut is small):

* :func:`rcb_partition`      -- Recursive Coordinate Bisection (Berger & Bokhari
  1987), the geometric partitioner used by Zoltan/Chaco for structured and
  mildly distorted meshes.  Zero imbalance by construction.
* :func:`graph_partition`    -- greedy region growing on the element adjacency
  graph followed by Kernighan-Lin style boundary refinement; the minimiser used
  when geometry is unavailable or unrepresentative.

Both return an ``(ne,)`` int array of entity -> owner plus diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "PartitionStats", "rcb_partition", "graph_partition",
    "element_adjacency", "partition_stats",
]


@dataclass(frozen=True)
class PartitionStats:
    nparts: int
    sizes: tuple[int, ...]
    imbalance: float          # (max - mean) / mean, 0.0 is perfect
    edge_cut: int             # adjacency edges crossing an owner boundary
    cut_fraction: float       # edge_cut / total adjacency edges
    max_cut_ratio: float      # edge_cut / (nparts - 1), interface-size proxy

    def as_dict(self) -> dict:
        return {
            "nparts": self.nparts,
            "sizes": list(self.sizes),
            "imbalance": self.imbalance,
            "edge_cut": self.edge_cut,
            "cut_fraction": self.cut_fraction,
        }


def _split_counts(n: int, nparts: int) -> list[int]:
    """Split ``n`` items into ``nparts`` counts differing by at most one."""
    base, rem = divmod(n, nparts)
    return [base + (1 if p < rem else 0) for p in range(nparts)]


def rcb_partition(points: np.ndarray, nparts: int) -> np.ndarray:
    """Recursive coordinate bisection on ``(n, dim)`` entity coordinates."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2:
        raise ValueError("points must be (n, dim)")
    n = pts.shape[0]
    if not 1 <= nparts <= n:
        raise ValueError(f"require 1 <= nparts <= n (got {nparts}, n={n})")
    owner = np.full(n, -1, dtype=np.int64)
    _rcb_rec(pts, np.arange(n), np.arange(nparts), owner)
    if (owner < 0).any():  # pragma: no cover - defensive
        raise RuntimeError("RCB left unassigned entities")
    return owner


def _rcb_rec(pts, idx, parts, owner) -> None:
    if parts.size == 0:
        return
    if parts.size == 1:
        owner[idx] = parts[0]
        return
    p = pts[idx]
    lo, hi = p.min(axis=0), p.max(axis=0)
    axis = int(np.argmax(hi - lo))
    order = np.argsort(p[:, axis], kind="stable")
    idx = idx[order]
    left_parts = parts.size // 2
    split = int(round(idx.size * left_parts / parts.size))
    split = min(max(split, left_parts), idx.size - (parts.size - left_parts))
    _rcb_rec(pts, idx[:split], parts[:left_parts], owner)
    _rcb_rec(pts, idx[split:], parts[left_parts:], owner)


def element_adjacency(conn: np.ndarray) -> list[set[int]]:
    """Face-sharing adjacency of quadrilateral elements from a (ne, 4) table."""
    conn = np.asarray(conn, dtype=np.int64)
    ne = conn.shape[0]
    adj: list[set[int]] = [set() for _ in range(ne)]
    seen: dict[tuple, list[int]] = {}
    for e in range(ne):
        nodes = conn[e]
        for a, b in ((nodes[0], nodes[1]), (nodes[1], nodes[3]),
                     (nodes[3], nodes[2]), (nodes[2], nodes[0])):
            key = (int(min(a, b)), int(max(a, b)))
            lst = seen.setdefault(key, [])
            for other in lst:
                adj[e].add(other)
                adj[other].add(e)
            lst.append(e)
    return adj


def graph_partition(adjacency: Sequence[set[int]], nparts: int,
                    *, refine_passes: int = 6) -> np.ndarray:
    """Greedy region-growing partition with Kernighan-Lin boundary refinement."""
    ne = len(adjacency)
    if not 1 <= nparts <= ne:
        raise ValueError(f"require 1 <= nparts <= ne (got {nparts}, ne={ne})")
    caps = _split_counts(ne, nparts)
    owner = np.full(ne, -1, dtype=np.int64)

    order = sorted(range(ne), key=lambda e: (-len(adjacency[e]), e))
    seeds: list[int] = []
    for cand in order:                       # spread seeds over the graph
        if len(seeds) == nparts:
            break
        if all(cand not in adjacency[s] and cand != s for s in seeds):
            seeds.append(cand)
    for cand in order:                       # top-up if graph is dense
        if len(seeds) == nparts:
            break
        if cand not in seeds:
            seeds.append(cand)

    loads = [0] * nparts
    frontier: list[list[int]] = []
    for p, s in enumerate(seeds):
        owner[s] = p
        loads[p] += 1
        frontier.append([s])
    while any(frontier):
        progress = False
        for p in range(nparts):
            f = frontier[p]
            if not f or loads[p] >= caps[p]:
                continue
            e = f.pop(0)
            progress = True
            for nb in sorted(adjacency[e]):
                if owner[nb] < 0 and loads[p] < caps[p]:
                    owner[nb] = p
                    loads[p] += 1
                    f.append(nb)
        if not progress:
            break
    for e in range(ne):                      # any isolated remainder
        if owner[e] < 0:
            p = int(np.argmin(loads))
            owner[e] = p
            loads[p] += 1

    # Kernighan-Lin style: move boundary entities when the cut strictly drops
    for _ in range(refine_passes):
        improved = 0
        for e in range(ne):
            own = int(owner[e])
            cut_own = sum(1 for nb in adjacency[e] if owner[nb] != own)
            if cut_own == 0:
                continue
            best_p, best_gain, best_net = own, 0, 0
            for p in range(nparts):
                if p == own:
                    continue
                cut_p = sum(1 for nb in adjacency[e] if owner[nb] != p)
                gain = cut_own - cut_p
                net = loads[own] - loads[p]      # >0 => p is lighter
                if gain > best_gain or (gain == best_gain and net > best_net + 1):
                    best_p, best_gain, best_net = p, gain, net
            if best_p != own:
                owner[e] = best_p
                loads[own] -= 1
                loads[best_p] += 1
                improved += 1
        if improved == 0:
            break
    return owner


def partition_stats(adjacency: Sequence[set[int]], owner) -> PartitionStats:
    owner = np.asarray(owner, dtype=np.int64)
    ne = owner.size
    nparts = int(owner.max()) + 1 if ne else 0
    sizes = tuple(int((owner == p).sum()) for p in range(nparts))
    mean = ne / nparts if nparts else 0.0
    imbalance = (max(sizes) - mean) / mean if nparts else 0.0
    cut = 0
    total = 0
    for e in range(ne):
        for nb in adjacency[e]:
            if nb > e:
                total += 1
                if owner[nb] != owner[e]:
                    cut += 1
    return PartitionStats(
        nparts=nparts,
        sizes=sizes,
        imbalance=float(imbalance),
        edge_cut=cut,
        cut_fraction=(cut / total) if total else 0.0,
        max_cut_ratio=(cut / max(1, nparts - 1)) if nparts > 1 else 0.0,
    )