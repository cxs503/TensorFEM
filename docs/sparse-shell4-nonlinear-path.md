# Sparse Shell4 nonlinear path qualification

The finite-rotation layered Shell4 increment and Crisfield arc-length solvers
accept an explicit `linear_solver="superlu"` option.  The default remains
`"dense"`; selecting SuperLU never silently falls back to dense and requires
the optional NumPy/SciPy CPU stack.

The sparse route assembles the reduced element tangent directly in COO form,
constructs the bordered arc-length matrix without densification, and records
factorization/solve time, matrix/factor nonzeros, estimated storage, right-hand
side count, and backward residual in the caller-supplied diagnostics list.
Each Newton tangent currently changes after an iteration, so factorization is
reused for its associated right-hand side only.  The adapter supports multiple
right-hand sides, but this arc formulation does not invent extra solves merely
to claim reuse.

Free Shell4 drilling rotations contain gauge modes for which dense execution
uses an SVD minimum-norm solve.  SuperLU has no equivalent rank-deficient
minimum-norm mode.  The sparse path therefore applies an explicit diagonal
gauge penalty only to free drilling rows, by default `1e-12` times the largest
matrix entry.  Its actual dimensional value is emitted as
`drilling_regularization` for every factorization.  It can be disabled by
setting `sparse_drilling_regularization=0`, in which case a singular
factorization fails closed.

## Real path evidence

The following CPU measurements used the same residual-stressed imperfect
marine panel, dimensional arc metric, normalized bordered equations, and
`ds = 0.02 t`.  They are qualification observations, not general speed claims.

| Mesh/path | Dense | SuperLU | load difference | displacement difference | sparse storage observation |
|---|---:|---:|---:|---:|---:|
| 2x2, 3 accepted points | 7.140 s | 6.825 s | 2.14e-13 relative | 1.90e-9 relative | sparse matrix 14,876 B vs dense bordered 14,792 B; factors 20,944 B |
| 4x4, 2 accepted points | 19.033 s | 19.172 s | 5.21e-11 relative | 6.40e-8 relative | sparse matrix 60,988 B vs dense bordered 133,128 B; factors 152,876 B |

Both routes converged with identical accepted-point counts.  Sparse linear
backward residuals were at most `3.10e-14` (2x2) and `2.66e-14` (4x4), and all
reported load/displacement differences are far below the 1% qualification
limit.  The 2x2 terminal recoverable energies were respectively
`29.846832711574528 J` and `29.846832711580973 J` (relative difference
`2.16e-13`), with zero hardening energy in both. Byte-for-byte state SHA-256
values differ because the two linear
algebra routes produce floating-point differences; this is reported rather
than misrepresented as state identity.  Both paths remain elastic in this
short prefix, so their material history invariants are physically identical.

No speedup is claimed: the 4x4 measurement is slightly slower, and these
prefixes are too small to establish scaling.  Long peak/post-peak paths,
plastic-history equivalence, process peak RSS, and larger meshes remain open
qualification work.
