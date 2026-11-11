import sys, time, os
sys.path.insert(0, 'src')
import torch, numpy as np
from tensorfem.parallel.comm import spawn
from tensorfem.parallel.poisson import DistPoisson


def work(comm, nx):
    torch.set_num_threads(1)
    d = DistPoisson(comm, nx, nparts=comm.size)
    u = torch.zeros(d.nown + d.nghost, dtype=torch.float64)
    ts = []
    for _ in range(40):
        t0 = time.perf_counter()
        d.halo(u)
        ts.append(time.perf_counter() - t0)
    sends = {j: np.ones(d._plan[j]['send_loc'].size) for j in d.neighbors}
    sizes = {j: d._plan[j]['recv_n'] for j in d.neighbors}
    raw = []
    for _ in range(40):
        t0 = time.perf_counter()
        comm.exchange(sends, sizes)
        raw.append(time.perf_counter() - t0)
    # per-iteration CG cost
    b = d.assemble_rhs()
    d.cg(b, iters=3)
    cg = []
    for _ in range(5):
        t0 = time.perf_counter()
        d.cg(b, iters=20)
        cg.append(time.perf_counter() - t0)
    return dict(rank=comm.rank, halo_us=round(min(ts) * 1e6, 1),
                raw_us=round(min(raw) * 1e6, 1), nbr=len(d.neighbors),
                per_iter_ms=round(min(cg) / 20 * 1e3, 2))


if __name__ == '__main__':
    nx = int(sys.argv[1]) if len(sys.argv) > 1 else 320
    for p in (1, 2, 4, 8):
        out, w = spawn(p, work, nx)
        print(f"ranks={p} halo_us={[o['halo_us'] for o in out]} "
              f"raw_us={[o['raw_us'] for o in out]} "
              f"per_iter_ms={[o['per_iter_ms'] for o in out]} "
              f"nbr={[o['nbr'] for o in out]}", flush=True)