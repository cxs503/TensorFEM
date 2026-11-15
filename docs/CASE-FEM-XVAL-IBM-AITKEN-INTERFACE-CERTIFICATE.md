# CASE-FEM-XVAL-IBM-AITKEN-INTERFACE-CERTIFICATE — what the sub-iteration tolerance actually certifies

Anchor: `XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION` (joconnor22/LIFE recipe) · repo TensorFEM
Benchmark: `benchmarks/xval_aitken_interface_certificate.py` →
`benchmarks/xval_aitken_interface_certificate_artifacts/{evidence.json,summary.md,*.png}`
Tests: `tests/test_aitken_interface_certificate.py` (13 tests)
Acceptance: cross-validation error < 15 % (owner standard); engineering target < 3 % where a closed form exists.

## 1 What is being validated

The v6 cross-validation of this anchor checked the *physics* of the implicit IBM + Aitken loop
(kernel, epsilon conservation, beam inertia, added mass, strong coupling). This round checks the
**convergence certificate** of the sub-iteration loop of
`tensorfem.ibm_aitken_implicit.ImplicitIBMFSI`: the residual that the loop compares against
`sub_tol` and the residual the Aitken Δ² estimate is built from.

The relaxed vector is a corotational-beam state: per node two translations (length) and one
rotation (radians). The historical criterion was

    res_legacy = ||r|| / (ref_length * sqrt(n_free))                (pre-fix, kept as "legacy_isotropic")

which divides a *rotation* residual in radians by a *length*. Dimensionally that is rad/m; as a
certificate, a rotation-only residual of `m` passes when `m <= sub_tol * ref_length * sqrt(n_free/n_rot)`
— on the v6 submerged beam (`ref_length = 72`, 30 free DOFs of which 10 are rotations) the angular
allowance is `sub_tol * 124.7 = 1.247e-2 rad` for `sub_tol = 1e-4`, i.e. 125x the requested angular
accuracy (the pure `ref_length` factor applies to a single rotational DOF, the extra `sqrt(3)` is the
RMS over the mixed DOF counts). The DOF-aware criterion is

    res_scaled = || W r || / sqrt(n_free),  W = diag(1/ref_length for lengths, 1/ref_angle for angles)

with the same `W` applied inside the Δ² dot products (`AitkenRelaxation.omega_for(..., scale=W)`),
because Δ² sums `r·dr` over components: with mixed units that sum adds length² to rad², so the
relaxation factor — hence the iteration — depends on the arbitrary length unit.

## 2 What was done

1. `dof_scale_vector(body, ref_length, ref_angle)` + `CorotationalBeam2D.dof_kind`
   (0 = length, 1 = angle); bodies without a declaration fall back to all-length (no DOF is
   given a looser weight than dimensional analysis allows).
2. `ImplicitIBMFSI(..., ref_angle=1.0, residual_measure="dof_scaled"|"legacy_isotropic")`;
   the step record now carries `residual_translation` / `residual_rotation`, so the certificate
   can be audited per DOF type instead of as one number.
3. `AitkenRelaxation.omega_for(..., scale=None)` — optional per-component weights; a *uniform*
   weight is a no-op, so every single-unit caller (plate driver, modal beam, scalar probes)
   keeps its exact behaviour.
4. `legacy_isotropic` reproduces the pre-fix expression **bit for bit** (test asserts equality with
   the archived formula), so all archived runs stay reproducible.

## 3 Results

Submerged cantilever (v6 setting: 160x120 lattice, dt = 5e-3, rho_f = 0.2, ref_length = 72,
30 free DOFs of which 10 are rotations), `sub_tol = 0.0001`:

| quantity | value |
|---|---|
| residual reported by the loop (mean) | 4.066e-05 |
| achieved rotation residual (max) | 7.238e-03 rad = 4.147e-01 deg |
| achieved translation residual (max) | 8.598e-05 (dimensionless) |
| rotation share of the residual | 98.9% |
| angular allowance, legacy | 1.247e-02 rad |
| angular allowance, DOF-scaled | 1.000e-04 rad |

Sub-tolerance sweep (deviation measured against an over-converged `sub_tol = 1e-12` reference):

| rho_f | measure | sub_tol | sub-it mean | max angle [rad] | max transl | tip dev / amplitude |
|---|---|---|---|---|---|---|
| 0.2 | legacy_isotropic | 0.01 | 1.00 | 7.24e-03 | 8.60e-05 | 5.36e-09 |
| 0.2 | dof_scaled | 0.01 | 1.00 | 7.24e-03 | 8.60e-05 | 5.36e-09 |
| 0.2 | legacy_isotropic | 0.001 | 1.00 | 7.24e-03 | 8.60e-05 | 5.36e-09 |
| 0.2 | dof_scaled | 0.001 | 1.96 | 1.59e-03 | 8.85e-06 | 3.00e-10 |
| 0.2 | legacy_isotropic | 0.0001 | 1.00 | 7.24e-03 | 8.60e-05 | 5.36e-09 |
| 0.2 | dof_scaled | 0.0001 | 2.00 | 1.99e-10 | 3.59e-12 | 5.60e-15 |
| 0.2 | legacy_isotropic | 1e-06 | 2.00 | 1.99e-10 | 3.59e-12 | 5.60e-15 |
| 0.2 | dof_scaled | 1e-06 | 2.00 | 1.99e-10 | 3.59e-12 | 5.60e-15 |
| 1 | legacy_isotropic | 0.01 | 1.00 | 7.24e-03 | 8.60e-05 | 2.68e-08 |
| 1 | dof_scaled | 0.01 | 1.00 | 7.24e-03 | 8.60e-05 | 2.68e-08 |
| 1 | legacy_isotropic | 0.001 | 1.00 | 7.24e-03 | 8.60e-05 | 2.68e-08 |
| 1 | dof_scaled | 0.001 | 1.96 | 1.59e-03 | 8.85e-06 | 1.50e-09 |
| 1 | legacy_isotropic | 0.0001 | 1.00 | 7.24e-03 | 8.60e-05 | 2.68e-08 |
| 1 | dof_scaled | 0.0001 | 2.00 | 9.97e-10 | 1.79e-11 | 9.00e-15 |
| 1 | legacy_isotropic | 1e-06 | 2.00 | 9.97e-10 | 1.79e-11 | 9.00e-15 |
| 1 | dof_scaled | 1e-06 | 2.00 | 9.97e-10 | 1.79e-11 | 9.00e-15 |

Backward compatibility at the v6 setting (`sub_tol = 1e-6`): 25/25 steps,
sub-iteration records identical = True, tip difference
0.000e+00 (0.000e+00 relative). The archived v6
submerged-beam numbers are therefore unaffected.

Coupled references (must not move):

* plate added mass (Lamb, single-DOF plate driver — untouched by construction, measured anyway):
  ratio 0.714370 vs reference 0.790569 → **9.64%**
  error (1500 steps; the v6 record at 1800 steps reported 7.48 %, the difference is
  the shorter period record, not the change).
* strong coupling m_a/m_s = 12: fixed ω = 1 diverges after 0 steps,
  Aitken runs 40/40 steps with 0 MAXIT hits.

Δ² unit invariance — same physics expressed in three length units (production `AitkenRelaxation`,
2-DOF mixed fixed point, `ref_length = 32`):

| gains | variant | max ω deviation between units | iteration spread | invariant |
|---|---|---|---|---|
| added_mass_gain_1.05 | legacy_unweighted | 2.000e+00 | 0 | **NO** |
| added_mass_gain_1.05 | dof_scaled | 1.021e-10 | 0 | yes |
| added_mass_gain_1.20 | legacy_unweighted | 2.000e+00 | 0 | **NO** |
| added_mass_gain_1.20 | dof_scaled | 1.516e-12 | 0 | yes |
| mild_gain_0.50 | legacy_unweighted | 0.000e+00 | 10 | yes |
| mild_gain_0.50 | dof_scaled | 0.000e+00 | 10 | yes |

Why the defect does not move the answer: the measured spectral radius of the beam interface map is
2.097e-07 (median |λ| 8.885e-24; the rotation share of the initial residual is
87.0%). The trial state does not feed back into the interface load at that time
level (marker velocities come from the committed state), so the sub-iteration is a *re-solve* that
converges in two sweeps. Under-relaxation is then inert, and a wrong certificate cannot corrupt the
fixed point — it can only mis-report how accurately it was reached. This is reported as a
limitation of the driver, not as a pass.

## 4 Error judgement (owner standard)

| key quantity | error | < 3 % ? |
|---|---|---|
| `legacy_isotropic` vs archived pre-fix residual formula | 0 (exact) | **yes** |
| tip record, legacy vs new default measure, v6 setting | 0.0e+00 | **yes** |
| max solution deviation caused by the certificate (all sub_tol, both rho_f) | 2.7e-08 | **yes** |
| plate added-mass ratio vs Lamb (independent reference) | 9.64% | **no** (record-length; v6 7.48 % at 1800 steps, within 15 %) |
| Δ² ω, DOF-scaled, across a 10⁶ change of length unit | 1e-10 | **yes** (machine noise) |
| Δ² ω, unweighted, across the same unit change | up to 2.0 (sign flip) | **no** — the defect |
| angular accuracy certified vs achieved, legacy | factor 125 too loose | **no** — the defect |

## 5 Caveats / limitations

* The LIFE executable is still not run (no paired LIFE run); every reference here is closed-form,
  or the archived v6 measurement.
* The Δ² unit-invariance evidence uses a controlled 2-DOF fixed point driven by the production
  `AitkenRelaxation`, not a live multi-DOF FSI case: the only production driver with mixed DOFs
  (`ImplicitIBMFSI`) has a measured interface-map spectral radius of ~2e-07, so it cannot
  exhibit the failure — the estimate never leaves the ω = 1 clip there. The defect is latent, and
  the fix is cheap insurance for drivers where the map does feed back (e.g. once the beam driver
  uses sub-iteration-consistent marker velocities).
* `ref_angle = 1 rad` is a convention, not a physical scale: a user who wants angular errors
  weighted by a characteristic length can pass `ref_angle` accordingly. Documented, not hidden.
* Scope: `fsi_lbm_strong_coupling.AitkenRelaxation` (modal coordinates, all DOFs are lengths) and
  `ibm_aitken_added_mass.IBMImplicitFSI` (one transverse DOF) were inspected and left unchanged —
  their residuals are single-unit, so the mixed-unit argument does not apply.

## 6 Reproduction

```
python -m pytest tests/test_aitken_interface_certificate.py -q   # 13 passed, 5 s
python benchmarks/xval_aitken_interface_certificate.py           # 7/7 checks, ~95 s single core
```
