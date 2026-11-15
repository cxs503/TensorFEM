# AUDIT — GAP-PP-3-ICE-STRUCT-PLASTIC-SET (adversarial verification of the merged plate-plasticity module)

Anchor: 板格低温钢弹塑性 + 永久挠度 (TensorFEM, branch `agent/m21466`).
Under review: `src/tensorfem/ice_plate_plasticity.py` (966 lines, merged `691eeabe`, m21011)
and `tests/test_ice_plate_plasticity.py`. Ledger status at review time: **open** —
implementation merged, acceptance never independently verified.

Independence limitation: the FE numbers use the same assembler as the module
(`layered_shell4_plasticity`); only the *references* and the *acceptance drivers*
were rebuilt independently. The 3-grid lateral matrix is NOT RUN (see below).

## Severity-ranked findings

### F1 (blocking, false pass) — step-quantised acceptance criteria
`verify_elastic_plastic_transition` returned the load *step* at which yield was first
seen, `step = 1.5*analytical/50 = 3%` of the reference; `verify_plastic_limit_load`
returned a step of a 2x scan with `n_steps=80` (2.5%). Because the returned load is
always a multiple of the step, the reported error can never exceed half a step (1.5%
and 1.25%) regardless of FE accuracy. The "<5%" / "<10%" criteria were satisfied by the
step grid, not by the FE. Measured (pre-fix, uniaxial, 8x4): fe = 362.1 MPa vs 355 MPa
-> reported 2.0% = exactly `1.5*355*34/50`.
Fix: bisection to `gap/2**refine`; `load_resolution` reported on the result
(default 14 refinements -> 3%/2^14 ≈ 2e-4%).

### F2 (blocking, false pass) — fallback reports perfect agreement on FE failure
`fe_limit = max_load_factor if max_load_factor > 0 else ilyushin_limit`. If no step
converged (broken solve, singular tangent, wrong BC) `fe_limit` became *exactly* the
reference -> relative_error = 0 -> passed = True. A total FE failure scored perfect.
Fix: fallback removed; no limit point now returns fe_limit=nan, relative_error=inf,
passed=False, applicable=False, plus a reason string.

### F3 (blocking, wrong reference) — lateral limit-pressure formula
`p = 4 Mp/b^2 (2b/a + a/b)` is refuted twice: (i) at a/b=1 it gives 12 Mp/b^2 versus the
classic simply-supported square-slab yield-line value 24 Mp/L^2 (50% low); (ii) it
*grows* with a/b (72 Mp/b^2 at a/b=8, unbounded as a->inf) whereas a longer plate cannot
carry more pressure — the exact one-way strip limit is 8 Mp/b^2. The 10% criterion could
only be met accidentally where the two errors cancel (a/b=2).
Fix: `yield_line_limit_pressure()` — virtual-work minimisation over two admissible
mechanism families with the ridge term retained, validated on three classical limits.

### F4 (blocking, wrong reference) — elastic first-yield pressure
The lateral transition reference was a guess `alpha = 0.75*aspect^2` / `0.75` with no
source; the correct square-plate coefficient is 0.2874 (nu=0.3), so the guess is 2.6x
too high and the "analytical yield load" 2.6x too low.
Fix: `navier_centre_moments()` + `elastic_first_yield_pressure()`, cross-checked against
Timoshenko & Woinowsky-Krieger tables.

### F5 (blocking, MOCK) — `quantity="limit_load"` in the convergence study
`verify_mesh_convergence` set `value = target_load`: the same hardcoded number for every
mesh, so all relative errors were identically zero. Independently, `passed` tested
`relative_errors[-1] < 0.05`, which is zero *by construction* (finest grid is the
reference) -> the pass condition was trivially true for any input, and
`convergence_rate` could never be computed.
Fix: limit load computed per grid; pass criterion on successive-grid change; rate from
the grid refinement ratio.

### F6 (overstated claim) — "arc-length continuation"
The limit-load docstring claimed arc-length continuation through the limit point; the
code did monotone load control and stopped at the first Newton failure (no arc-length,
no branch switching, no post-limit path). Docstring corrected; note added that
load control can fail slightly before the true limit load (biases fe_limit low).

### F7 (non-blocking) — dead / incoherent code
`prev_load_factor` unused; the transition bisection re-started from a *yielded* committed
state and then loaded it *down* (wrong incremental problem once yield is path dependent)
— fixed by keeping the last elastic state; `stresses` unused; the NR loop commits a
plastic state not corresponding to the committed displacement (staggered update: still
converges, degrades the rate).

### F8 (test quality) — tests executed code, did not assert invariants
Acceptance-relevant tests only asserted positivity (`fe_limit > 0`, `values[0] > 0`) or
`fe_limit >= 0.95*reference`; none asserted <5%/<10%.
`test_lateral_limit_scales_with_thickness` asserted only the t^2 scaling, which the wrong
formula also satisfies. `test_convergence_three_meshes` used a 2x1 grid on which the FE
solve fails (values 0) and was already failing on main.
Fix: 8 new invariant tests (`TestReferenceInvariants`, `TestNoFalsePass`).

## Verification evidence (programmatic)

Elastic reference (Navier double-sine, nu=0.3) vs Timoshenko tables, M/(p b^2) at centre:

| a x b | table | this work | error |
|---|---|---|---|
| 1x1 | 0.0479 | 0.047886 | 0.02 % |
| 2x1 | 0.1017 | 0.101680 | 0.02 % |
| 50x1 (strip 0.125) | 0.1250 | 0.124880 | 0.10 % |

Lateral collapse pressure (sigma_y = 355 MPa, t = 20 mm, b = 1 m), pre- vs post-audit:

| a/b | audited (MPa) | pre-audit (MPa) | pre/audited | strip 8Mp/b^2 |
|---|---|---|---|---|
| 1 | 0.8520 | 0.4260 | 0.50 | 0.2840 |
| 1.25 | 0.6942 | 0.4047 | 0.58 | 0.2840 |
| 1.5 | 0.6024 | 0.4023 | 0.67 | 0.2840 |
| 2 | 0.5020 | 0.4260 | 0.85 | 0.2840 |
| 2.5 | 0.4489 | 0.4686 | 1.04 | 0.2840 |
| 3 | 0.4164 | 0.5207 | 1.25 | 0.2840 |
| 4 | 0.3787 | 0.6390 | 1.69 | 0.2840 |
| 6 | 0.3442 | 0.8993 | 2.61 | 0.2840 |
| 8 | 0.3281 | 1.1715 | 3.57 | 0.2840 |
| 12 | 0.3127 | 1.7277 | 5.53 | 0.2840 |
| 20 | 0.3009 | 2.8542 | 9.49 | 0.2840 |

Anchors: audited a/b=1 equals the classic 24 Mp/L^2 to <0.01%; a/b=200 -> 8.07 Mp/b^2 vs
the exact strip 8 (+0.9%); monotone decreasing in a/b; p_limit > p_first_yield for every
aspect (1.7-2.2x), as plasticity theory requires.

Pre-fix FE number (uniaxial, 8x4, sigma_y=355 MPa): fe_yield = 362.1 MPa -> 2.0%, exactly
one load step from the reference. This is the exhibit for F1, not evidence of 2% accuracy.

## Checks not run (unknown, NOT "passed")
* 3-grid lateral matrix (transition + limit + permanent deflection at 8x4/12x6/16x8):
  submitted as compute job **cj3469**, still queued after 25 min (47 jobs in flight).
  Local execution exceeds the <10 min short-test window and a detached background run
  was refused by the local-compute guard. Status: **not_run**.
* Post-fix resolved transition/limit errors on any grid: **not_run**; the resolution
  guarantee is a property of the code path and is exposed as `load_resolution`, but no
  post-fix converged FE number was produced.
* Permanent-deflection correctness beyond the trivial invariant (a virgin elastic state
  gives exactly 0.000e+00 m — verified): no unloading-hysteresis or large-deflection
  reference check. **not_run**.
* `ilyushin_combined_limit` interaction term: spot-checked only (Q=1 at uniaxial yield,
  Q<1 elastic); not verified against a coded reference. **not_run**.

## Disposition
**Request changes; the acceptance claim must not be accepted.**
F1-F5 are repaired in the working tree and the repaired references are validated against
classical limits (8 new tests green, plus the 21 earlier non-FE tests). The anchor's
criteria (transition <5%, limit <10%, 3-grid convergence) are now *measurable* but not
yet *met*: the resolved FE numbers are pending on compute job cj3469. The ledger entry
must stay `open`/`in_progress`, not `done`.
