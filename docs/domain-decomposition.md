# Domain decomposition prototype

`domain_decomposition.py` provides deterministic contiguous and graph-aware
ownership, explicit owned/halo/interface maps, and overlapping additive
Schwarz preconditioning for SPD sparse systems. Local matrices are exact dense
Cholesky solves; the global Krylov iteration remains matrix-free and sparse.

The single-process implementation deliberately simulates several subdomains.
This makes decomposition results deterministic and testable without a cluster.
`distributed_sum` is the optional `torch.distributed` collective adapter and
fails closed when no process group has been initialized. Assignment of
subdomains to ranks and communication scheduling remain application concerns.

## Qualification contract

- partition labels must be contiguous, nonempty, disjoint in ownership and
  cover every global DOF exactly once;
- halo values share a single global unknown, so interface continuity is exact;
- local matrices and the global operator must be SPD;
- CG non-convergence is reported, never silently accepted;
- contiguous and graph partitions with 1, 2, 3 and 5 subdomains reproduce the
  global dense solution to relative error below `1e-9` (well below 3%).

Run `pytest -q tests/test_domain_decomposition.py`. Large topology/operator
smoke tests are opt-in and excluded from pytest:

```bash
PYTHONPATH=src python examples/domain_decomposition_benchmark.py --ndof 100000
PYTHONPATH=src python examples/domain_decomposition_benchmark.py --ndof 1000000
```

This prototype does **not** claim MPI halo scheduling, coarse-grid correction,
METIS quality, distributed autograd, GPU-aware collectives, or scalability.
