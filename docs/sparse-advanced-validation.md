# Advanced sparse solver validation

The opt-in module `sparse_advanced` adds portable Jacobi preconditioning,
restarted left-preconditioned GMRES, and multiple-right-hand-side load-case
solves. Dense, COO, CSR and matrix-free operators use the same public PyTorch
implementation and preserve input dtype/device.

Automated qualification includes an SPD diagonal system with condition number
`1e8`, a symmetric-indefinite system checked against `torch.linalg.solve`, and
three simultaneous load cases. All solution errors must be strictly below 3%;
the normal observed error is roundoff. A CUDA/CPU consistency test runs when a
CUDA device is available.

Jacobi is used because it is portable across qualified PyTorch CPU/CUDA sparse
backends. Incomplete Cholesky is deliberately not claimed: public PyTorch does
not currently provide a portable sparse IC factorization. Likewise this module
does not claim sparse direct factorization, distributed assembly, domain
decomposition, or block Krylov acceleration; multiple RHS currently reuse the
operator and solve each load vector independently.
