# Shell4 sparse iterative scaling qualification

TensorFEM's finite-rotation Shell4 Newton and arc-length paths now accept
`linear_solver="ilu_gmres"`. The route retains the reduced COO tangent and
bordered arc matrix, constructs a SciPy ILU preconditioner, and solves with
SciPy's native restarted GMRES while independently checking the true residual
as a PyTorch tensor. The default remains `dense`.
Missing or broken NumPy/SciPy support fails closed; there is no implicit dense
or SuperLU fallback.

Controls are explicit: `ilu_drop_tolerance`, `ilu_fill_factor`,
`krylov_tolerance`, and `krylov_max_iterations`. Diagnostics expose ILU
nonzeros/storage, factor and solve timings, preconditioner refresh/reuse
decisions, Krylov iterations, and true unpreconditioned residuals. The same audited
drilling-gauge stabilization used by the qualified SuperLU route is applied.

The full nonlinear integration test traces three accepted Shell4 arc points
and agrees with dense load and displacement to `1e-6` relative tolerance.
This verifies that iterative solves are used inside predictor and bordered
Newton correctors rather than only in a detached linear benchmark.
In addition, a residual-stressed imperfect 2x2 marine panel completed its
initial-equilibrium relaxation and first accepted dimensional arc point with
both sparse routes: SuperLU and ILU-GMRES produced respectively
`28,604.075840514 N` and `28,604.075840176 N`, with displacement norms
`0.009338385852114 m` and `0.009338385852099 m`. ILU-GMRES used 45 total
Krylov iterations across all predictor/corrector solves.

The same full-flow check was then run on the larger 8x8 panel. Both routes
accepted the first point: SuperLU produced `17,253.182558440 N` and displacement
norm `0.008720980686686 m` in 52.84 s; ILU-GMRES produced
`17,253.182558449 N` and `0.008720980687167 m` in 53.62 s, using 65 Krylov
iterations. The load and displacement differences are `5.30e-13` and
`5.51e-11` relative. This confirms large-grid full-flow correctness, while the
1.5% longer elapsed time again prevents a speedup claim.

## Real panel-tangent scaling

`scripts/run_shell_sparse_scaling.py --meshes 8 12` assembles real imperfect
marine-panel tangents and solves a manufactured system whose exact solution is
known. Measurements below use ILU drop tolerance `1e-4`, fill factor `10`, and
GMRES tolerance `1e-8`.

| metric | 8x8 | 12x12 |
|---|---:|---:|
| free DOFs / matrix nnz | 444 / 19,842 | 952 / 45,270 |
| GMRES iterations | 11 | 16 |
| true relative residual | 1.89e-11 | 2.91e-9 |
| solution relative error | 3.05e-8 | 4.07e-5 |
| ILU matrix + factor storage | 859,056 B | 2,373,984 B |
| SuperLU matrix + factor storage | 1,052,388 B | 3,055,716 B |
| dense matrix storage | 1,577,088 B | 7,250,432 B |
| ILU factor + solve time | 0.0591 s | 0.1327 s |
| SuperLU factor + solve time | 0.0069 s | 0.0215 s |

Both solution errors are below the 1% gate. At 12x12, ILU-GMRES reduces
matrix-plus-factor storage by 22.3% relative to SuperLU and by 67.3% relative
to a dense tangent. This establishes the required larger-grid memory
advantage. It does **not** establish a time advantage: SuperLU is substantially
faster at both measured sizes. Global element tangent assembly (4.68 s and
9.56 s) also dominates either linear solve at these scales.

Consequently ILU-GMRES is qualified as an optional memory-oriented path, not
as the default solver and not as a general acceleration claim. Larger models,
adaptive ILU controls, reusable symbolic structure, and stronger Schwarz/AMG
preconditioners remain necessary before claiming an industrial time crossover.

## Native iterative execution and refresh study

The next scaling increment keeps GMRES, sparse matrix actions, and ILU
applications inside SciPy for the entire linear solve. This removes one
Python/Torch/NumPy boundary crossing per Krylov iteration while preserving a
PyTorch tensor result and an independently recomputed true residual. The
optional dependency contract is unchanged.

On 12x12, native ILU-GMRES used 20 iterations, 0.0385 s factorization and
0.0171 s solve time, versus 0.0203 s and 0.0025 s for SuperLU. On 20x20
(2,544 free DOFs and 127,230 nonzeros), it used 43 iterations, 0.1271 s
factorization and 0.0733 s solve time, versus 0.0973 s and 0.0068 s for
SuperLU. The respective solution errors were `1.38e-6` and `3.04e-5`, safely
below 1%. At 20x20, ILU matrix-plus-factor storage is 9,286,080 B versus
12,652,176 B for SuperLU and 51,775,488 B dense.

Refreshing an ILU preconditioner every fourth solve was also tested on the
complete 8x8 first arc point. It reduced fresh ILU constructions from six to
three, but increased total Krylov iterations from 65 to 83 and changed elapsed
time from 53.62 s to 53.77 s. An inexact `1e-6` linear tolerance reduced
iterations to 57 but took 54.55 s. Both variants preserved the path far inside
the 1% gate, yet neither improved wall time.

No measured time crossover exists through 2,544 free DOFs. A coarse trend
extrapolation suggests that direct-factor fill would need models on the order
of tens of thousands of free DOFs before ILU-GMRES might cross SuperLU, but
this is explicitly not qualification evidence. The dominant measured
bottleneck is element tangent assembly: 9.98 s at 12x12 and 26.90 s at 20x20,
compared with at most 0.20 s for either complete linear phase. Meaningful path
acceleration therefore requires analytic/AD element tangents, batched element
assembly, or tangent reuse—not additional Krylov tuning alone.
