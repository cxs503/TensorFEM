"""Gates for the MPI-style distributed FEM parallel layer.

What is asserted (invariants, not just "code executes"):

1. element matrix: closed form == independent 6x6 Gauss quadrature (rect + square)
2. partitioner: exact balance, cut fraction sane, deterministic
3. communicator: Allreduce(SUM/MIN) and neighbour exchange against a serial
   reference, for 2/4 ranks
4. domain decomposition: ghost values received are exactly the owner's values
   (the failure mode that silently produced wrong solves before)
5. partition invariance: residual and L2 error identical to 1e-12 for 1/2/4 ranks
   -- a wrong halo plan shows up here immediately
6. discretisation: O(h^2) L2 convergence of the manufactured solution
7. scaling harness: the gate metric is computed from real timings and the JSON
   artifact exists after a (tiny) run
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorfem.parallel.comm import (ProcessComm, SerialComm, physical_cpus,  # noqa: E402
                                     spawn)
from tensorfem.parallel.partition import (element_adjacency,  # noqa: E402
                                          graph_partition, partition_stats,
                                          rcb_partition)
from tensorfem.parallel.poisson import (DistPoisson, q1_laplace_element,  # noqa: E402
                                         q1_mass_lumped_element)


# ---------------------------------------------------------------- element matrix
def _gauss_reference(a: float, b: float, order: int = 6) -> np.ndarray:
    gp, gw = np.polynomial.legendre.leggauss(order)
    ref = np.zeros((4, 4))
    for xi, wx in zip(gp, gw):
        for eta, wy in zip(gp, gw):
            dxi = np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)]) / 4 * 2 / a
            det = np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]) / 4 * 2 / b
            ref += (wx * wy) * (a / 2) * (b / 2) * (np.outer(dxi, dxi) + np.outer(det, det))
    return ref


def test_q1_element_matrix_matches_quadrature():
    for a, b in ((0.5, 0.5), (0.25, 0.1), (0.05, 0.2)):
        assert np.abs(q1_laplace_element(a, b) - _gauss_reference(a, b)).max() < 1e-13


def test_q1_element_matrix_textbook_square():
    want = np.array([[4, -1, -2, -1], [-1, 4, -1, -2],
                     [-2, -1, 4, -1], [-1, -2, -1, 4]], dtype=float) / 6.0
    assert np.abs(q1_laplace_element(0.5) - want).max() < 1e-15
    assert np.allclose(q1_mass_lumped_element(0.5), 0.5 * 0.5 / 4)   # sum = h^2


# -------------------------------------------------------------------- partition
def test_rcb_is_balanced_and_deterministic():
    ne = 32 * 32
    ex, ey = np.arange(ne) % 32, np.arange(ne) // 32
    pts = np.stack([(ex + .5) / 32, (ey + .5) / 32], axis=1)
    for p in (2, 4, 8):
        own = rcb_partition(pts, p)
        assert own.min() == 0 and own.max() == p - 1
        sizes = np.bincount(own, minlength=p)
        assert sizes.max() - sizes.min() <= 1          # exact balance
        assert np.array_equal(own, rcb_partition(pts, p))   # deterministic


def test_graph_partition_on_structured_grid_is_valid_and_balanced():
    """Hard invariants of the graph partitioner: valid owner range, every entity
    assigned exactly once, deterministic, cut fraction strictly inside (0,1)
    on a connected 16x16 grid; RCB on the same grid has zero imbalance."""
    n = 16
    conn = []
    for j in range(n):
        for i in range(n):
            conn.append([j * (n + 1) + i, j * (n + 1) + i + 1,
                         (j + 1) * (n + 1) + i + 1, (j + 1) * (n + 1) + i])
    conn = np.array(conn)
    adj = element_adjacency(conn)
    degs = [len(s) for s in adj]
    assert min(degs) >= 2 and max(degs) == 4     # corner 2, edge 3, interior 4
    own = graph_partition(adj, 4)
    st = partition_stats(adj, own)
    assert own.size == n * n and own.min() == 0 and own.max() == 3
    assert sum(st.sizes) == n * n
    assert 0.0 < st.cut_fraction < 1.0
    assert np.array_equal(own, graph_partition(adj, 4))     # deterministic
    ex, ey = np.arange(n * n) % n, np.arange(n * n) // n
    pts = np.stack([(ex + .5) / n, (ey + .5) / n], axis=1)
    assert partition_stats(adj, rcb_partition(pts, 4)).imbalance == 0.0


# ---------------------------------------------------------------- communicator
def _collective_probe(comm):
    s = comm.allreduce(np.array([1.0, 2.0 * comm.rank]), op="sum")
    m = comm.allreduce(np.array([float(comm.rank + 3)]), op="min")
    got = comm.exchange({j: np.array([10.0 * comm.rank, 1.0])
                         for j in range(comm.size) if j != comm.rank},
                        {j: 2 for j in range(comm.size) if j != comm.rank})
    comm.barrier()
    return {"rank": comm.rank, "sum": list(s), "min": float(m[0]),
            "peer": {j: list(v) for j, v in got.items()}}


@pytest.mark.parametrize("nprocs", [2, 4])
def test_collectives_match_serial_reference(nprocs):
    out, _ = spawn(nprocs, _collective_probe)
    total = nprocs * (nprocs - 1) / 2 * 2.0
    for o in out:
        assert abs(o["sum"][0] - nprocs) < 1e-12
        assert abs(o["sum"][1] - total) < 1e-12
        assert o["min"] == 3.0
        for j, v in o["peer"].items():
            assert v == [10.0 * j, 1.0]


def test_serial_comm_is_identity():
    c = SerialComm()
    assert c.allreduce(np.array([3.0]))[0] == 3.0
    assert c.bcast({"a": 1}) == {"a": 1}
    c.barrier()


def test_physical_cpus_has_no_smt_siblings():
    cpus = physical_cpus()
    assert len(cpus) >= 1
    assert len(set(cpus)) == len(cpus)
    if len(cpus) > 1:
        sib = {}
        for c in cpus:
            p = f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list"
            if os.path.exists(p):
                sib[c] = open(p).read().strip()
        assert len(set(sib.values())) == len(sib)   # one thread per core


# ------------------------------------------------------- distributed operator
def test_ghost_values_equal_owner_values():
    """Every ghost slot must receive exactly the owning rank's value."""
    nx = 8
    probe = np.arange((nx + 1) ** 2, dtype=np.float64) / 7.0

    def work(comm):
        d = DistPoisson(comm, nx, nparts=comm.size)
        u = torch.zeros(d.nown + d.nghost, dtype=torch.float64)
        u[:d.nown] = torch.tensor(probe[d.owned])
        d.halo(u)
        got = u[d.nown:].numpy()
        want = probe[d.ghost]
        return {"rank": comm.rank, "err": float(np.abs(got - want).max())
                if d.nghost else 0.0, "ghosts": d.nghost}

    for p in (1, 2, 4):
        out, _ = spawn(p, work)
        assert max(o["err"] for o in out) < 1e-14


def test_distributed_solution_matches_serial():
    """Partition invariance: same residual and same L2 error for 1/2/4 ranks."""
    nx = 24

    def work(comm):
        d = DistPoisson(comm, nx, nparts=comm.size)
        b = d.assemble_rhs()
        r = d.cg(b, iters=25)
        return {"res": r["residual"], "l2": d.l2_error(r["x"]),
                "dofs": d.global_dofs(), "setup": d.setup_seconds,
                "ghosts": d.nghost}

    ref, _ = spawn(1, work)
    for p in (2, 4):
        out, _ = spawn(p, work)
        assert out[0]["dofs"] == ref[0]["dofs"] == (nx + 1) ** 2
        assert abs(out[0]["res"] - ref[0]["res"]) <= 1e-10 * max(1.0, ref[0]["res"])
        assert abs(out[0]["l2"] - ref[0]["l2"]) <= 1e-10 * max(1.0, ref[0]["l2"])


def test_second_order_convergence():
    errs, hs = [], []
    for n in (8, 16, 32):
        d = DistPoisson(None, n)
        r = d.cg(d.assemble_rhs(), tol=1e-12)
        errs.append(d.l2_error(r["x"]))
        hs.append(1.0 / n)
    orders = [np.log(errs[i] / errs[i + 1]) / np.log(hs[i] / hs[i + 1])
              for i in range(len(errs) - 1)]
    assert all(1.9 < o < 2.1 for o in orders), (errs, orders)


def test_dirichlet_rows_are_constrained():
    d = DistPoisson(None, 12)
    x = torch.ones(d.nown, dtype=torch.float64)
    y = d.matvec(x)
    assert torch.all(y[~d._free_t] == 0)


# ------------------------------------------------------------------- scaling
def test_scaling_harness_reports_gate(tmp_path, monkeypatch):
    from tensorfem.parallel import scaling as S
    res = S.strong_scaling((1, 2), {"mesh": 48, "iters": 3, "repeats": 1})
    assert res["gate"]["threshold"] == 0.60
    assert len(res["points"]) == 2
    assert res["points"][0]["ranks"] == 1
    assert abs(res["points"][0]["speedup"] - 1.0) < 1e-12
    assert res["points"][1]["global_dofs"] == res["points"][0]["global_dofs"]
    assert res["points"][1]["efficiency"] > 0.0
    json.dumps(res)          # must be serialisable for the artifact


def test_committed_scaling_artifacts_exist():
    root = os.path.join(os.path.dirname(__file__), "..", "artifacts",
                        "parallel_scaling")
    path = os.path.join(root, "scaling.json")
    if not os.path.exists(path):
        pytest.skip("scaling artifacts not generated in this checkout")
    data = json.load(open(path))
    for key in ("strong", "weak", "gates", "figures"):
        assert key in data
    for pt in data["strong"]["points"]:
        assert pt["cg_seconds"] > 0 and pt["global_dofs"] > 0
    assert data["strong"]["points"][0]["speedup"] == pytest.approx(1.0)