"""Strong / weak scaling harness for the distributed FEM path.

Protocol (predeclared before any timing was inspected -- see
docs/parallel-domain-decomposition.md):

* kernel     : one distributed preconditioned-CG solve (``DistPoisson.cg``) of
               the same global problem on the same mesh, with a **fixed
               iteration budget**.  Partition invariance of CG was verified
               first (identical residual and L2 error to 12 significant digits
               for 1/2/4/8 ranks), so a fixed budget means every rank count
               performs bit-comparable work.
* strong     : total problem size fixed (nx = ny = mesh), ranks = 1,2,4,8.
               speedup S(p) = median T(1) / median T(p); efficiency S(p)/p.
* weak       : work per rank fixed (nx = ny = mesh_per_rank), total mesh grows
               with p; report T(p)/T(1) (ideal 1.0) and efficiency.
* timing     : only the CG loop is timed (mesh build, partition and comm-plan
               setup are reported separately as ``setup_s``).  Each point is
               the median over ``repeats`` independent fresh solves.
* exclusions : none.  No point is dropped; the full grid is reported.
* gate       : parallel efficiency >= 0.60 at 4 ranks (strong scaling).

All numbers written to JSON carry the exact configuration used so a run can be
reproduced by ``python benchmarks/parallel_scaling.py``.
"""
from __future__ import annotations

import statistics
import time

import numpy as np
import torch

from .comm import backend_available, spawn
from .poisson import DistPoisson

D = torch.float64

DEFAULT_STRONG = {"mesh": 640, "iters": 30, "repeats": 3}
DEFAULT_WEAK = {"mesh_per_rank": 320, "iters": 30, "repeats": 3}
DEFAULT_RANKS = (1, 2, 4, 8)


def _time_cg(d: DistPoisson, b, iters: int, repeats: int) -> tuple[float, list]:
    """Median wall time of ``repeats`` fresh CG solves with a fixed budget."""
    d.matvec(b)                                   # warm-up: page in / JIT
    times = []
    last = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        last = d.cg(b, iters=iters)
        times.append(time.perf_counter() - t0)
    return statistics.median(times), times


def _time_parts(d: DistPoisson, b, iters: int) -> dict:
    """Break the iteration down: local matvec, halo exchange, collective."""
    x = torch.zeros_like(b)
    d.matvec(x)                                    # warm-up
    t0 = time.perf_counter()
    for _ in range(5):
        d.matvec(x)
    t_matvec = (time.perf_counter() - t0) / 5.0

    u = torch.zeros(d.nown + d.nghost, dtype=D)
    d.halo(u)
    t0 = time.perf_counter()
    for _ in range(5):
        d.halo(u)
    t_halo = (time.perf_counter() - t0) / 5.0

    val = np.array([1.0])
    d.comm.allreduce(val, op="sum")
    t0 = time.perf_counter()
    for _ in range(5):
        d.comm.allreduce(val, op="sum")
    t_allred = (time.perf_counter() - t0) / 5.0
    return {
        "t_matvec_s": t_matvec,
        "t_halo_s": t_halo,
        "t_allreduce_s": t_allred,
        "per_iter_comm_s": t_halo + 2.0 * t_allred,
    }


def rank_work(comm, cfg: dict) -> dict:
    """Body executed on every rank; returns that rank's measurements."""
    torch.set_num_threads(1)
    nx = int(cfg["mesh"])
    ny = int(cfg.get("mesh_y", nx))
    nparts = int(cfg.get("nparts", comm.size))
    d = DistPoisson(comm, nx, ny, nparts=nparts,
                    partitioner=cfg.get("partitioner", "rcb"))
    b = d.assemble_rhs()
    iters = int(cfg["iters"])
    med, times = _time_cg(d, b, iters, int(cfg.get("repeats", 3)))
    parts = _time_parts(d, b, iters) if cfg.get("breakdown", True) else {}
    res = d.cg(b, iters=iters)
    out = {
        "rank": comm.rank,
        "nx": nx,
        "ny": ny,
        "nown": d.nown,
        "nghost": d.nghost,
        "global_dofs": int(d.global_dofs()),
        "neighbors": len(d.neighbors),
        "halo_bytes": d.halo_bytes,
        "setup_s": d.setup_seconds,
        "cg_seconds": med,
        "cg_seconds_samples": times,
        "iters": iters,
        "residual": res["residual"],
        "l2_error": d.l2_error(res["x"]),
    }
    out.update(parts)
    if comm.rank == 0 and d.stats is not None:
        out["partition"] = d.stats.as_dict()
    return out


def _run(ranks: int, cfg: dict) -> dict:
    out, wall = spawn(ranks, rank_work, cfg)
    r0 = out[0]
    worst = max(o["cg_seconds"] for o in out)
    return {
        "ranks": ranks,
        "mesh": r0["nx"],
        "mesh_y": r0["ny"],
        "global_dofs": r0["global_dofs"],
        "global_elements": r0["nx"] * r0["ny"],
        "cg_seconds": r0["cg_seconds"],
        "cg_seconds_worst_rank": worst,
        "cg_seconds_samples": r0["cg_seconds_samples"],
        "wall_seconds_including_spawn": wall,
        "setup_s": r0["setup_s"],
        "iters": r0["iters"],
        "residual": r0["residual"],
        "l2_error": r0["l2_error"],
        "max_neighbors": max(o["neighbors"] for o in out),
        "halo_bytes_per_rank": sorted({o["halo_bytes"] for o in out})[-1],
        "ghosts": [o["nghost"] for o in out],
        "owned": [o["nown"] for o in out],
        "t_matvec_s": r0.get("t_matvec_s"),
        "t_halo_s": r0.get("t_halo_s"),
        "t_allreduce_s": r0.get("t_allreduce_s"),
        "per_iter_comm_s": r0.get("per_iter_comm_s", 0.0),
        "partition": r0.get("partition"),
    }


def strong_scaling(ranks=DEFAULT_RANKS, cfg: dict | None = None) -> dict:
    cfg = dict(DEFAULT_STRONG if cfg is None else cfg)
    points = [_run(int(p), cfg) for p in ranks]
    t1 = points[0]["cg_seconds"]
    for pt in points:
        pt["speedup"] = t1 / pt["cg_seconds"]
        pt["efficiency"] = pt["speedup"] / pt["ranks"]
        pt["seconds_per_iter"] = pt["cg_seconds"] / pt["iters"]
    return {
        "study": "strong",
        "config": {**cfg, "ranks": list(ranks)},
        "backend": backend_available(),
        "points": points,
        "gate": {"metric": "efficiency_at_4_ranks", "threshold": 0.60,
                 "value": _eff_at(points, 4),
                 "passed": _eff_at(points, 4) >= 0.60},
    }


def weak_scaling(ranks=DEFAULT_RANKS, cfg: dict | None = None) -> dict:
    cfg = dict(DEFAULT_WEAK if cfg is None else cfg)
    base = int(cfg["mesh_per_rank"])
    points = []
    for p in ranks:
        # keep work per rank constant: p slabs of (base x base) cells
        c = dict(cfg, mesh=base * int(p), mesh_y=base)
        points.append(_run(int(p), c))
    t1 = points[0]["cg_seconds"]
    for pt in points:
        pt["speedup"] = t1 / pt["cg_seconds"]          # ideal 1.0 (time flat)
        pt["efficiency"] = pt["speedup"] / 1.0
        pt["seconds_per_iter"] = pt["cg_seconds"] / pt["iters"]
        pt["work_ratio_vs_p1"] = (pt["global_dofs"] / points[0]["global_dofs"])
    return {
        "study": "weak",
        "config": {**cfg, "ranks": list(ranks)},
        "backend": backend_available(),
        "points": points,
        "gate": {"metric": "weak_efficiency_at_4_ranks", "threshold": 0.60,
                 "value": _eff_at(points, 4), "passed": _eff_at(points, 4) >= 0.60},
    }


def _eff_at(points, ranks: int) -> float:
    for pt in points:
        if pt["ranks"] == ranks:
            return float(pt["efficiency"])
    return float("nan")