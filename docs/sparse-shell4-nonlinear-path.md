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

## Yielding-path qualification (v0.39)

The reproducible runner
`scripts/run_sparse_shell4_plastic_qualification.py` drives a three-layer,
finite-rotation Shell4 facet through 80 accepted arc-length points.  The final
state has yielded at every integration point, so this comparison exercises
trial/commit history and the consistent plastic tangent rather than only an
elastic prefix.

| Quantity | dense | SuperLU | relative difference |
|---|---:|---:|---:|
| accepted points | 80 | 80 | 0 |
| load factor | 7.719914400896420 | 7.719914400896421 | 1.15e-16 |
| displacement norm | 0.1238170125666957 | 0.1238170125666961 | 3.03e-15 |
| recoverable energy | 0.4308711361111206 J | 0.4308711361111226 J | 4.64e-15 |
| hardening energy | 0.4013992076224942 J | 0.4013992076224965 J | 5.81e-15 |
| sum of equivalent plastic strain | 1.015320107085858 | 1.015320107085861 | 3.06e-15 |
| plastic-strain tensor norm | 0.3591806137340720 | 0.3591806137340731 | 2.94e-15 |
| yielded fraction | 1.0 | 1.0 | 0 |

Both histories have monotonically nondecreasing equivalent plastic strain.
The maximum SuperLU backward residual is `3.07e-16`.  State SHA-256 values are
not byte-identical due to the floating-point differences shown above; physical
history invariants pass the 1% gate by more than twelve orders of magnitude.

Independent CPU processes measured 50.80 s for dense and 54.91 s for SuperLU
inside the qualification region. External `/usr/bin/time -v` peak RSS was
710,252 KiB and 734,168 KiB respectively. SuperLU performed 197 numerical
factorizations. At this four-free-DOF size its maximum reported sparse matrix
and factor storage estimates are 324 B and 408 B, versus a 200 B dense 5x5
bordered matrix. Thus this case qualifies nonlinear plastic-path equivalence,
but explicitly **fails to demonstrate either speed or memory improvement**.
Sparse performance qualification requires substantially larger yielding
models and symbolic/numerical factor reuse across compatible tangent updates.
