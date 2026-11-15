# XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION - interface convergence certificate

generated 2026-11-15T12:34:16Z  |  repo TensorFEM  |  tolerance 15%

## Submerged beam (lattice 160x120, ref_length = 72.0 lattice units, 30 free DOFs, 10 of them rotations)

* sub_tol = 0.0001; reported residual mean 4.066e-05
* achieved angular residual max 7.238e-03 rad (4.147e-01 deg)
* achieved translation residual max 8.598e-05 (dimensionless)
* rotation share of the raw residual 98.9%
* legacy angular allowance at this sub_tol 1.247e-02 rad vs 1.000e-04 rad for the DOF-scaled measure

## Tolerance sweep

| rho_f | measure | sub_tol | sub-it mean | max angle [rad] | tip dev / amplitude |
|---|---|---|---|---|---|
| 0.2 | legacy_isotropic | 0.01 | 1.00 | 7.24e-03 | 5.36e-09 |
| 0.2 | dof_scaled | 0.01 | 1.00 | 7.24e-03 | 5.36e-09 |
| 0.2 | legacy_isotropic | 0.001 | 1.00 | 7.24e-03 | 5.36e-09 |
| 0.2 | dof_scaled | 0.001 | 1.96 | 1.59e-03 | 3.00e-10 |
| 0.2 | legacy_isotropic | 0.0001 | 1.00 | 7.24e-03 | 5.36e-09 |
| 0.2 | dof_scaled | 0.0001 | 2.00 | 1.99e-10 | 5.60e-15 |
| 0.2 | legacy_isotropic | 1e-06 | 2.00 | 1.99e-10 | 5.60e-15 |
| 0.2 | dof_scaled | 1e-06 | 2.00 | 1.99e-10 | 5.60e-15 |
| 1 | legacy_isotropic | 0.01 | 1.00 | 7.24e-03 | 2.68e-08 |
| 1 | dof_scaled | 0.01 | 1.00 | 7.24e-03 | 2.68e-08 |
| 1 | legacy_isotropic | 0.001 | 1.00 | 7.24e-03 | 2.68e-08 |
| 1 | dof_scaled | 0.001 | 1.96 | 1.59e-03 | 1.50e-09 |
| 1 | legacy_isotropic | 0.0001 | 1.00 | 7.24e-03 | 2.68e-08 |
| 1 | dof_scaled | 0.0001 | 2.00 | 9.97e-10 | 9.00e-15 |
| 1 | legacy_isotropic | 1e-06 | 2.00 | 9.97e-10 | 9.00e-15 |
| 1 | dof_scaled | 1e-06 | 2.00 | 9.97e-10 | 9.00e-15 |

## Backward compatibility (v6 setting, sub_tol = 1e-6)

* steps 25 / 25; sub-iteration records identical: True
* tip difference legacy vs default measure: 0.000e+00 (0.000e+00 relative)
* angular residual: legacy 1.994e-10 vs default 1.994e-10

## Coupled references (must not move)

* plate added mass (Lamb, single-DOF driver): ratio measured 0.714370 vs reference 0.790569, error 9.6386% (v6 archived 7.48%)
* strong coupling m_a/m_s = 12: fixed omega diverged after 0 steps = True; Aitken 40/40 steps, 0 MAXIT

## Delta^2 unit invariance (production AitkenRelaxation)

| gains | variant | max omega deviation between units | iteration spread | invariant |
|---|---|---|---|---|
| added_mass_gain_1.05 | legacy_unweighted | 2.000e+00 | 0 | NO |
| added_mass_gain_1.05 | dof_scaled | 1.021e-10 | 0 | yes |
| added_mass_gain_1.20 | legacy_unweighted | 2.000e+00 | 0 | NO |
| added_mass_gain_1.20 | dof_scaled | 1.516e-12 | 0 | yes |
| mild_gain_0.50 | legacy_unweighted | 0.000e+00 | 10 | yes |
| mild_gain_0.50 | dof_scaled | 0.000e+00 | 10 | yes |

## Why the certificate defect does not move the answer

* measured spectral radius of the beam interface map 2.097e-07 (median eigenvalue 8.885e-24), rotation share of the initial residual 87.0%
* the trial state does not feed back into the interface load at this time level, so the sub-iteration is a re-solve that converges in two sweeps; the acceptance criterion then mis-certifies the angular accuracy but the fixed point is already reached.
