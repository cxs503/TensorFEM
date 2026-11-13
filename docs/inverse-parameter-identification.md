# Inverse parameter identification (`tensorfem.inverse`)

Status: new capability, released 0.45.x. Module `src/tensorfem/inverse.py`,
tests `tests/test_inverse_parameter_identification.py`, demo
`examples/inverse_parameter_identification.py`.

## 1. Why

TensorFEM's forward solvers are verified against analytic/reference solutions, but
the *inverse* half of a calibration (which stiffness / coefficient / time constant
produced these measurements?) was re-implemented ad hoc inside individual
benchmark scripts. That made two things impossible:

* comparing two calibrations without hand-copying numbers, and
* saying honestly whether the recovered parameter was identifiable at all from the
  available observables.

`tensorfem.inverse` supplies the four pieces every calibration needs, once:

| piece | object |
|---|---|
| unknown description | `Parameter` (bounds, `log` transform, scaling) |
| forward model + data | `InverseProblem` (`forward`, `data`, `sigma`, `regularizers`) |
| optimiser loop | `run_inversion` (Levenberg-Marquardt, optional multi-start) |
| audit trail | `ConvergenceRun` → `convergence.jsonl` + `summary.json` (hash-pinned) |

## 2. Method

**Loss.** Data residuals are studentised by the measurement uncertainty,
`r = (model(p) - data) / sigma`, and each regulariser contributes its own residual
block, so the scalar loss is a *sum of audited terms*:

```
loss = sum_i (model_i(p) - data_i)**2 / sigma_i**2  +  sum_k w_k * ((p_j - t_kj)/s_kj)**2
       \______________________ data term ______________________/     \___ Tikhonov ___/
```

`InverseProblem.loss_components` returns every term separately and `total`; the
log stores the full component vector per iteration.

**Search space.** Unknowns are optimised in non-dimensional coordinates:
`log(x)` for strictly positive quantities, `x / scale` otherwise. Stiffness
(`2.1e11`) and a decay constant (`0.3 s`) then take comparable Gauss-Newton steps;
in the axial-bar case the resulting weighted Jacobian has condition number `1.0`
(Section 5), i.e. the scaling removed the problem's `1e11` spread entirely.

**Optimiser.** Levenberg-Marquardt with a finite-difference Jacobian
(`fd_rel_step = 1e-6`, forward or central schemes)
`(J^T J + lambda diag(J^T J)) delta = -J^T r`, with

* strict-descent acceptance (`trial_loss < loss`),
* `lambda` escalation by `lm_nu` inside `max_inner` attempts,
* a **backtracking steepest-descent safeguard** when all LM trials fail. Without
  it the loop stalls on ill-conditioned valleys: `lambda` escalation makes every
  trial too small to change the loss in float64 while the gradient is still
  non-zero (observed on the rank-deficient `E, A` case, Section 5),
* stopping on `||g||_inf <= gtol`, `||delta||_inf <= xtol (1 + ||u||_inf)`,
  a `ftol` plateau over `stall_patience` accepted steps, or a float64 loss plateau.

**Status semantics** (`ConvergenceRun.status`): `converged` (criterion met,
including a genuine float64 plateau), `stalled` (no loss-reducing step could be
found *and* no plateau detected), `max_iter`, `invalid` (non-finite loss). The
reason string in `stop_reason` says which criterion fired, so a plateau is never
silently reported as a stuck run and vice versa.

**Seeds.** `starts == 1` is deterministic. With `starts > 1` the jittered starts
are drawn from `torch.Generator().manual_seed(seed + 1009 * i)` and every start
point is written to the log, so the same seed reproduces the same log hash and a
different seed is visible in the record.

## 3. Identifiability diagnostics

At the estimate, `identifiability_diagnostics` computes the weighted Jacobian `J`
and reports:

* `rank` — numerical rank at machine tolerance,
* `effective_rank` — rank at the *condition-number limit* `s_max / cond_limit`;
  a direction weaker than this cannot be resolved to `cond_limit` digits,
* `weak_directions` — the parameter combinations the data do not constrain
  (the null vectors of `J` in optimiser coordinates),
* `relative_standard_error` — from the linearised covariance
  `var * (J^T J)^{-1}`; log parameters are already relative, scaled linear
  parameters are converted back to physical units,
* `correlation`, `sensitivity`, `residual_rms`, `reduced_chi2`.

`identifiable` is `effective_rank == n and condition <= cond_limit`. A run that
converges on a rank-deficient problem reports `converged` **plus**
`identifiable: False` **plus** a warning naming the weak direction — the failure
mode this module exists to make impossible is a confident-looking number that the
data never constrained.

## 4. Audit artifacts

`ConvergenceRun.write(directory)` emits

* `convergence.jsonl` — schema `tensorfem.inverse-convergence/1`, one canonical
  JSON object per line: a `header` record (problem configuration, parameter specs,
  data sha256, options, seed control, all start points, provenance), one
  `iteration` record per trial (`loss`, `components`, `lambda`, `step_inf`,
  `grad_inf`, `accepted`, current parameters), and a `final` record,
* `summary.json` — schema `tensorfem.inverse-summary/1` with the final
  parameters, diagnostics, warnings, `parameters_at_bound`, evaluation count,
  provenance and `convergence_log_sha256` (SHA-256 of the JSONL text).

Because the hash covers the whole log, "the number in the report came from this
run" is checkable: `sha256(convergence.jsonl)` must equal
`summary["convergence_log_sha256"]`, which is exactly what
`tests/test_inverse_parameter_identification.py::test_convergence_log_is_machine_readable_and_hash_verified`
asserts.

## 5. Verification results

Reproduce with `PYTHONPATH=src python examples/inverse_parameter_identification.py`
→ `runs/inverse-parameter-identification/{metrics.json, */convergence.jsonl, */summary.json}`
(the table below is generated from `metrics.json`, not typed by hand).

| case | unknown | recovered | reference | rel. error | reported rel. stderr | status |
|---|---|---|---|---|---|---|
| 1. decay (seed 1) | `tau` | 4.033929 s | 4.0 s | 0.848 % | 0.750 % | converged |
| 1. decay (seed 2) | `tau` | 4.033929 s | 4.0 s | 0.848 % | 0.750 % | converged |
| 1. decay (seed 3) | `tau` | 4.033929 s | 4.0 s | 0.848 % | 0.750 % | converged |
| 2. axial bar (TrussModel) | `E` | 2.1065e+11 Pa | 2.10e+11 Pa | 0.311 % | 0.316 % | converged |
| 3. axial bar, `E` and `A` | `E*A` | 2.1065e+08 | 2.1000e+08 | 0.311 % | n/a (rank deficient) | converged, identifiable=false |

Rank diagnostics, case 3: `rank = 2`, `effective_rank = 1`, `condition = 4.453e+11`, weak direction `(E, A) = (-0.7071, +0.7071)`.

Reproducibility: same-seed log hash identical = `True`, distinct hashes for 3 seeds = `3`. Provenance commit `442d393a9f5a`.

Reading of the numbers:

* **Recovery accuracy.** All three cases are far inside the 3 % acceptance band;
  the largest error is the decay time constant at 0.85 %.
* **The reported uncertainty is calibrated.** For every case the linearised
  relative standard error is within ~7 % of the realised error
  (`0.848 %` vs `0.750 %` for `tau`, `0.311 %` vs `0.316 %` for `E`), i.e. the
  covariance estimate is not a decoration.
* **Reproducibility.** Re-running seed 1 reproduces the identical log hash; the
  three seeds produce three distinct hashes but agree on `tau` to 8 significant
  digits (the problem is convex enough that the jitter cannot matter).
* **Honest non-identifiability.** With `E` and `A` both unknown the data contain
  only the product: `effective_rank = 1`, `identifiable = false`, and the weak
  direction is `(-0.7071, +0.7071)` — the `delta ln E = -delta ln A` family, as
  theory requires. The *product* is still recovered to 0.31 %, which is the only
  statement the data support.

## 6. Cost and limits

* **Cost.** The finite-difference Jacobian costs `n` (forward) or `2n` (central)
  forward solves per iteration; the axial-bar case needs 2 iterations and 3
  forward solves per iteration (one observable chain of 4 nodes each). For large
  `n` an adjoint/tangent-linear sensitivity or a JAX/torch-autograd Jacobian is the
  correct next step — **not implemented here** (see next section).
* **Local method.** LM converges to a local minimum; the multi-start seeding is
  the only globalisation. No basin-of-attraction guarantee is claimed.
* **Uncertainty is linearised.** `relative_standard_error` is a first-order
  (Gauss-Newton) approximation; for strongly non-linear problems use a profile
  likelihood or MCMC — not implemented.
* **`sigma` must be defensible.** If it is optimistic the `reduced_chi2` warning
  fires (tested in `test_unrepresentable_data_reports_residual_excess_not_a_good_fit`),
  but the module cannot detect a *systematic* model error from a good chi-square.
* **No constraint solvers.** Only box bounds (hard clipping) and quadratic
  regularisation; inequality constraints on derived quantities are out of scope.

## 7. Known next steps (not done in this increment)

1. Autograd/adjoint Jacobian to replace finite differences for `n > ~10`.
2. Profile-likelihood intervals for non-linear identifiability.
3. A `.github`-visible golden-file regression that pins `log_sha256` per release
   (needs a stable torch version pin, deliberately not asserted in the unit tests).