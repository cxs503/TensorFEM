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

## Required next backend

A bounded next step is a true sparse factorization or industrial-grade
preconditioner (for example sparse LU/LDL, ILU, or AMG) with explicit drilling
nullspace handling.  SciPy is not installed in the project environment, so
`scipy.sparse.linalg.spsolve`/`splu` could not be evaluated in this run and
must remain an optional-backend experiment rather than a new mandatory
dependency.  Qualification must compare accepted load, displacement,
equilibrium residual, memory, and wall time against the dense reference on
both 4x4 and 8x8 real steps.
