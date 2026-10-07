# Sparse industrial core validation

The module provides deterministic node/component numbering, Dirichlet and
linear multi-point constraint elimination, differentiable vectorised COO/CSR
assembly, and matrix-free reduced preconditioned conjugate gradients.  The
solver fails closed for non-SPD curvature and non-convergence.

Automated tests compare sparse assembly and CG against dense PyTorch results.
The axial bar benchmark uses the analytical displacement `FL/(EA)` and requires
strictly less than 3% relative error (normally roundoff). Nonzero prescribed
displacements and an affine MPC are checked against direct reduced energy
minimisation.

Run the opt-in scale test with:

```bash
PYTHONPATH=src python examples/sparse_100k_benchmark.py --dofs 100000 \
  --output sparse-100k.json
```

The script uses 100,000 independent ground springs with analytical `u=f/k=1`
and records assembly/solve times, nonzeros, iterations, residual and analytical
error. It is excluded from default pytest because runtime depends on
CPU/GPU and PyTorch sparse backends.

Current limitations: CG requires SPD systems; chained MPCs, Lagrange multiplier
saddle systems, sparse direct factorisation, MINRES/GMRES, domain decomposition
and distributed assembly are not provided. COO/CSR index creation can dominate
small models, and PyTorch sparse autograd/backend coverage depends on device.
