# Coupled million-DOF structural sparse benchmark

The benchmark is a rectangular nearest-neighbour spring structure with one
scalar displacement per free node and a fixed frame around its boundary. Its
SPD five-point stiffness has about five million nonzeros at one million DOFs;
it is explicitly not a diagonal throughput test.

The load is manufactured from four exact discrete sine eigenmodes. This gives
an analytical nodal displacement field while exercising matrix assembly,
sparse matrix-vector products and CG across several distinct eigenvalues. The
runner reports exact COO tensor storage, assembly and solve times, iteration
count, residual and displacement error. It fails closed on non-convergence or
error at or above 3%.

CI uses a 1,280-DOF case and independently compares small matrices with dense
Cholesky solutions. The large case is opt-in:

```bash
PYTHONPATH=src python examples/million_dof_structural_benchmark.py
PYTHONPATH=src python examples/million_dof_structural_benchmark.py --nx 316 --ny 317
```

The benchmark characterizes a coupled structural topology and portable PyTorch
sparse execution. It does not claim a full vector-elastic finite element,
distributed-memory scaling, or wall-clock comparability across machines.
