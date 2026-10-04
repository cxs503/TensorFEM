# Panel sparse-solver qualification

## Current status

The sparse Shell4 assembly is storage- and action-qualified, but the iterative
nonlinear solve is **not qualified** for the 4x4 or 8x8 compressed-panel path.
This boundary is intentional: reduced storage is not evidence of a reliable or
faster nonlinear solver.

## Reproduced evidence

At the undeformed panel state, the reduced tangent spectra were measured in
float64:

| mesh | free DOFs | largest singular value | smallest singular value | condition estimate |
|---|---:|---:|---:|---:|
| 4x4 | 128 | 7.854e9 | 2.239e-2 | 3.51e11 |
| 8x8 | 444 | 8.793e9 | 7.002e-3 | 1.26e12 |

Both systems contain one near-null global drilling-rotation gauge.  Their
membrane, bending, and drilling scales also span several orders of magnitude.
The existing Jacobi, node-block, and element-block preconditioners all failed
the real 4x4 first-step corrector; the respective accumulated Krylov counts in
the bounded reproduction were 527, 530, and 1411.

Symmetric diagonal coordinate equilibration reduced the linearized condition
estimate to about 1.33e5.  It solved the isolated 4x4 predictor in 114 GMRES
iterations with a 2.11e-11 relative residual.  This did not qualify the
nonlinear path: updated-tangent, drilling-nullspace projection, and bordered
arc-constraint elimination still exhausted 15 corrector iterations (about
3500 Krylov iterations and 32 seconds) without accepting the first 4x4 point.
The isolated 8x8 predictor reached only a 1.40e-7 relative residual after 600
iterations.

These experimental solver changes were therefore not retained in production
code.  No speedup or 4x4/8x8 sparse-path convergence claim is made.

## Optional SuperLU/ILU experiment

`tensorfem.sparse_direct` now provides a deliberately non-public, lazy adapter
for SciPy SuperLU (`splu`) and incomplete LU (`spilu`).  It does not add a
package dependency, does not appear in `tensorfem.__init__` or the stable API
manifest, and never falls back to a dense solve.  An explicit request fails
with `SparseBackendUnavailable` when NumPy/SciPy is absent, when a CUDA tensor
is supplied, or when the optional stack cannot be imported.  Invalid and
singular factorizations also fail closed.

The TensorFEM virtual environment has neither NumPy nor SciPy.  The following
experiment temporarily exposed the server's already-existing Hermes
NumPy/SciPy installation through `PYTHONPATH`; nothing was installed and that
environment is not a TensorFEM runtime dependency.  The matrices are real
undeformed finite-rotation Shell4 panel tangents.  The bordered matrices use
the production Crisfield corrector block structure (tangent, reference-load
column, displacement-increment row, and arc constraint scalar).  All values
are float64 on CPU:

| mesh | free DOFs | solve | SuperLU residual | relative difference from dense | dense time | SuperLU time |
|---|---:|---|---:|---:|---:|---:|
| 4x4 | 128 | predictor | 5.11e-15 | 3.42e-8 | 0.52 ms | first-call 159.7 ms |
| 4x4 | 129 | bordered corrector | 6.71e-15 | 3.79e-9 | 0.54 ms | 2.50 ms |
| 8x8 | 444 | predictor | 1.42e-14 | 6.68e-9 | 2.32 ms | 8.06 ms |
| 8x8 | 445 | bordered corrector | 1.41e-14 | 2.86e-9 | 2.01 ms | 8.19 ms |

The solution differences reflect the panels' condition estimates of
`3.51e11` and `1.26e12`; the independently evaluated residuals demonstrate
that both solvers satisfy their equations.  These are linear corrector
comparisons, not accepted nonlinear-path timing claims.

For 4x4, COO tangent storage is 114,768 bytes versus 131,072 bytes for the
reduced dense tangent; SuperLU has 12,551 combined L/U nonzeros.  For 8x8,
COO storage is 476,208 bytes versus 1,577,088 bytes (30.2%); SuperLU has
67,412 combined L/U nonzeros.  Factor fill must therefore be included in any
future memory claim instead of quoting input COO storage alone.

Default ILU controls (`drop_tol=1e-4`, `fill_factor=10`) produced stand-alone
relative residuals of 4.32e-3 (4x4) and 1.59e-2 (8x8).  This is unsuitable as
an exact corrector solve.  ILU remains only a candidate preconditioner whose
effect on true-residual GMRES and complete accepted paths must be qualified.

The direct backend is accurate but slower than dense at these mesh sizes, and
the default project environment cannot activate it.  It is retained as an
optional foundation; **no sparse speedup or nonlinear-path qualification is
claimed**.

## Required next backend work

The remaining step is integration into a complete path behind an explicit
solver selection, followed by accepted-load, displacement, equilibrium,
factor-memory, and end-to-end wall-time comparisons.  Industrial-grade ILU or
AMG also needs explicit drilling-nullspace handling.  Promotion requires a
measured advantage on larger meshes; availability and linear accuracy alone
are insufficient.
