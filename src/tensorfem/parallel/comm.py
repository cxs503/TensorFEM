"""Communication framework for distributed-memory FEM in TensorFEM.

Two interchangeable backends implement one ``Comm`` protocol:

* :class:`MPIComm`   -- wraps ``mpi4py`` (real MPI); used on an HPC cluster.
* :class:`ProcessComm` -- a reference implementation over ``multiprocessing``
  pipes with MPI semantics (rank/size, point-to-point sendrecv, recursive
  doubling allreduce, barrier).  Used when ``mpi4py`` is not installed so the
  parallel code path is testable and measurable anywhere.

* :class:`SerialComm` -- size == 1, every collective is a local no-op.

The point-to-point API is byte-buffer based (``numpy`` arrays) so that halo
exchange costs raw ``tobytes`` copies instead of pickled tensors.

Scaling note: ``ProcessComm.allreduce`` uses recursive doubling (log2(size)
point-to-point exchanges, one pipe per ordered rank pair), i.e. the same
algorithmic structure as an MPI reduce; it is *not* a central-hub gather, so
measured scaling is not an artefact of a broadcast-shaped collective.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from typing import Callable, Mapping, Sequence

import numpy as np

__all__ = [
    "Comm", "SerialComm", "ProcessComm", "MPIComm",
    "backend_available", "spawn",
]

_OP_SUM = "sum"
_OP_MAX = "max"
_OP_MIN = "min"


class Comm:
    """Abstract communicator (MPI subset needed by the FEM parallel path)."""

    rank: int = 0
    size: int = 1

    # -- identity -----------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    # -- point to point -----------------------------------------------------
    def sendrecv(self, send: np.ndarray | None, dest: int,
                 recv_nbytes: int, source: int) -> np.ndarray:
        raise NotImplementedError

    # -- collectives --------------------------------------------------------
    def allreduce(self, send: np.ndarray, op: str = _OP_SUM) -> np.ndarray:
        raise NotImplementedError

    def bcast(self, obj, root: int = 0):
        raise NotImplementedError

    def barrier(self) -> None:
        raise NotImplementedError


class SerialComm(Comm):
    """Single-rank communicator: all collectives are local identities."""

    rank = 0
    size = 1

    def sendrecv(self, send, dest, recv_nbytes, source):  # noqa: D102
        raise RuntimeError("SerialComm has no peers")

    def allreduce(self, send, op=_OP_SUM):  # noqa: D102
        return np.array(send)

    def bcast(self, obj, root=0):  # noqa: D102
        return obj

    def barrier(self) -> None:  # noqa: D102
        return None


# ---------------------------------------------------------------------------
# multiprocessing backend
# ---------------------------------------------------------------------------
_HDR = 8  # header bytes: (tag:int32, nbytes:int32)


def physical_cpus() -> list[int]:
    """One logical CPU per physical core, interleaved across sockets.

    SMT siblings must not host two ranks: on the 2-socket/32-core/2-thread dev
    node, unbound ranks collide on sibling threads and the resulting skew shows
    up as multi-ms waits inside every halo exchange.  Falls back to
    ``range(cpu_count())`` when the topology is unavailable.
    """
    n = os.cpu_count() or 1
    firsts: list[int] = []
    for c in range(n):
        path = f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list"
        try:
            head = open(path).read().strip().split(",")[0]
            first = int(head.split("-")[0])
        except OSError:
            first = c
        if first == c:
            firsts.append(c)
    if not firsts:
        return list(range(n))
    try:
        pkg = {}
        for c in firsts:
            path = f"/sys/devices/system/cpu/cpu{c}/topology/physical_package_id"
            pkg.setdefault(int(open(path).read().strip()), []).append(c)
        if len(pkg) <= 1:
            return firsts
        groups = [pkg[k] for k in sorted(pkg)]
        out: list[int] = []
        while any(groups):
            for g in groups:
                if g:
                    out.append(g.pop(0))
        return out
    except OSError:
        return firsts


def pin_to_cpu(cpu: int) -> bool:
    """Bind the current process to one CPU; returns True on success."""
    try:
        os.sched_setaffinity(0, {int(cpu)})
        return True
    except (AttributeError, OSError):  # pragma: no cover - non-Linux
        return False


def _pack(tag: int, arr: np.ndarray) -> bytes:
    a = np.ascontiguousarray(arr)
    return np.int32([tag, a.nbytes]).tobytes() + a.tobytes()


def _recv_bytes(conn) -> tuple[int, np.ndarray]:
    raw = conn.recv_bytes()
    tag, n = np.frombuffer(raw[:8], dtype=np.int32)
    return int(tag), np.frombuffer(raw[8:8 + int(n)], dtype=np.float64)


class ProcessComm(Comm):
    """MPI-semantics communicator over per-rank-pair duplex pipes."""

    def __init__(self, rank: int, size: int, conns: Mapping[int, object]):
        self.rank = rank
        self.size = size
        self._c = dict(conns)
        self._pool = None

    # -- helpers ------------------------------------------------------------
    def _peer(self, other: int):
        try:
            return self._c[other]
        except KeyError as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                f"rank {self.rank} has no pipe to rank {other} "
                f"(neighbours: {sorted(self._c)})") from exc

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
        for c in self._c.values():
            try:
                c.close()
            except Exception:  # noqa: BLE001 pragma: no cover
                pass
        self._c.clear()

    # -- point to point -----------------------------------------------------
    def sendrecv(self, send, dest, recv_nbytes, source):
        """Exchange one buffer with one peer.

        Deadlock-free ordering: within a rank pair the lower rank sends first,
        the higher rank receives first.  Pairs are independent (own pipe), so
        simultaneous exchanges with several neighbours cannot deadlock.
        """
        send = None if send is None else np.asarray(send, dtype=np.float64)
        tag = 7
        lower = self.rank < dest
        if lower:
            if send is not None:
                self._peer(dest).send_bytes(_pack(tag, send))
            raw = self._peer(source).recv_bytes() if recv_nbytes else b""
        else:
            raw = self._peer(source).recv_bytes() if recv_nbytes else b""
            if send is not None:
                self._peer(dest).send_bytes(_pack(tag, send))
        if not recv_nbytes:
            return np.zeros(0, dtype=np.float64)
        t, n = np.frombuffer(raw[:8], dtype=np.int32)
        return np.frombuffer(raw[8:8 + int(n)], dtype=np.float64).copy()

    def exchange(self, sends: Mapping[int, np.ndarray],
                 recv_sizes: Mapping[int, int]) -> dict[int, np.ndarray]:
        """Symmetric neighbour exchange (fast path used by halo update).

        Every pair (me, j) contributes ``sends[j]`` (may be empty) and expects
        ``recv_sizes[j]`` float64 values back.  Implemented as one sendrecv per
        neighbour with the even/odd rank ordering above.
        """
        peers = sorted(set(sends) | set(recv_sizes))
        if len(peers) <= 1:
            out: dict[int, np.ndarray] = {}
            for j in peers:
                out[j] = self.sendrecv(sends.get(j), j,
                                       int(recv_sizes.get(j, 0)) * 8, j)
            return out
        # Independent pipes => the round trips can overlap.  MPI posts all
        # sends/recvs at once, so we do the same with one thread per peer
        # (conn.send_bytes/recv_bytes release the GIL while blocked).
        if self._pool is None:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(max_workers=max(1, self.size - 1))
        pool = self._pool
        futs = {j: pool.submit(self.sendrecv, sends.get(j), j,
                               int(recv_sizes.get(j, 0)) * 8, j) for j in peers}
        return {j: f.result() for j, f in futs.items()}

    # -- collectives --------------------------------------------------------
    def allreduce(self, send, op=_OP_SUM):
        buf = np.array(send, dtype=np.float64).ravel()
        n = buf.size
        dist = 1
        while dist < self.size:
            partner = self.rank ^ dist
            recv = self.sendrecv(buf, partner, n * 8, partner)
            if op == _OP_SUM:
                buf = buf + recv
            elif op == _OP_MAX:
                buf = np.maximum(buf, recv)
            elif op == _OP_MIN:
                buf = np.minimum(buf, recv)
            else:  # pragma: no cover
                raise ValueError(f"unsupported op {op!r}")
            dist <<= 1
        return buf

    def bcast(self, obj, root=0):
        if self.size == 1:
            return obj
        if self.rank == root:
            for r in range(self.size):
                if r != root:
                    self._peer(r).send(_msg(obj))
        else:
            obj = self._peer(root).recv()
        return obj

    def barrier(self) -> None:
        self.allreduce(np.zeros(1), op=_OP_SUM)


def _msg(obj):
    return ("bcast", obj)


def spawn(nprocs: int, target: Callable, *args, **kwargs):
    """Run ``target(comm, *args, **kwargs)`` on ``nprocs`` ranks.

    Returns the list of per-rank return values ordered by rank.  With
    ``nprocs == 1`` it runs inline in the caller process (no fork at all), so
    the single-rank baseline pays zero framework cost.
    """
    if nprocs < 1:
        raise ValueError("nprocs must be >= 1")
    if nprocs == 1:
        return [target(SerialComm(), *args, **kwargs)], 0.0

    ctx = mp.get_context("fork")
    pipes: dict[tuple[int, int], tuple] = {}
    for i in range(nprocs):
        for j in range(i + 1, nprocs):
            a, b = ctx.Pipe(duplex=True)
            pipes[(i, j)] = (a, b)          # a: lower rank end, b: higher

    def conns_for(r):
        out = {}
        for (i, j), (a, b) in pipes.items():
            if r == i:
                out[j] = a
            elif r == j:
                out[i] = b
        return out

    results: dict[int, object] = {}
    errors: dict[int, str] = {}
    queues = [ctx.Queue(1) for _ in range(nprocs)]
    t0 = time.perf_counter()

    def _entry(r, q):
        # isolate BLAS/OpenMP: one thread per rank, scaling must come from MPI
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        try:
            import torch
            torch.set_num_threads(1)
        except Exception:  # noqa: BLE001 pragma: no cover
            pass
        if os.environ.get("TFEM_PARALLEL_AFFINITY", "1") != "0":
            cpus = physical_cpus()
            if cpus:
                pin_to_cpu(cpus[r % len(cpus)])
        comm = ProcessComm(r, nprocs, conns_for(r))
        try:
            q.put(("ok", target(comm, *args, **kwargs)))
        except BaseException as exc:  # noqa: BLE001
            import traceback
            q.put(("err", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
        finally:
            comm.close()

    procs = [ctx.Process(target=_entry, args=(r, queues[r]), daemon=False)
             for r in range(nprocs)]
    for p in procs:
        p.start()
    for r, p in enumerate(procs):
        p.join()
        try:
            status, payload = queues[r].get_nowait()
        except Exception:  # noqa: BLE001 - rank died before reporting
            status, payload = "err", f"rank exited with code {p.exitcode}"
        if status == "ok":
            results[r] = payload
        else:
            errors[r] = payload
    wall = time.perf_counter() - t0
    if errors:
        raise RuntimeError(
            "rank failure(s):\n" + "\n".join(f"[{r}] {e}"
                                             for r, e in sorted(errors.items())))
    return [results[r] for r in range(nprocs)], wall


class MPIComm(Comm):
    """mpi4py-backed communicator (production path on the cluster)."""

    def __init__(self, mpi_comm=None):
        try:
            from mpi4py import MPI  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("mpi4py is not installed") from exc
        self._MPI = MPI
        self._comm = mpi_comm if mpi_comm is not None else MPI.COMM_WORLD
        self.rank = self._comm.Get_rank()
        self.size = self._comm.Get_size()

    def sendrecv(self, send, dest, recv_nbytes, source):
        MPI = self._MPI
        sbuf = np.ascontiguousarray(send if send is not None else np.zeros(0))
        rbuf = np.empty(recv_nbytes // 8, dtype=np.float64)
        self._comm.Sendrecv(sbuf, dest=dest, recvbuf=rbuf, source=source)
        return rbuf

    def exchange(self, sends, recv_sizes):
        return {j: self.sendrecv(sends.get(j), j, int(recv_sizes.get(j, 0)) * 8, j)
                for j in sorted(set(sends) | set(recv_sizes))}

    def allreduce(self, send, op=_OP_SUM):
        MPI = self._MPI
        f = {_OP_SUM: MPI.SUM, _OP_MAX: MPI.MAX, _OP_MIN: MPI.MIN}[op]
        buf = np.array(send, dtype=np.float64).ravel()
        out = np.empty_like(buf)
        self._comm.Allreduce(buf, out, op=f)
        return out

    def bcast(self, obj, root=0):
        return self._comm.bcast(obj, root=root)

    def barrier(self):
        self._comm.Barrier()


def backend_available() -> dict:
    """Report which backends can be used on this machine."""
    info = {"mpi4py": False, "process": True, "cpu_count": os.cpu_count()}
    try:
        import mpi4py  # noqa: F401
        info["mpi4py"] = True
    except Exception:  # noqa: BLE001
        pass
    return info