# UNDEX shock chain -- Cole anchor, citation ledger and FEM closure (chain-3)

Task: `paper-exp-undex-shock-chain-3` (mission 15973), domain: structural response to underwater-explosion shock.

## 1. Contents

| artifact | content |
|---|---|
| `cole_anchor.json` | Cole (1948) constants, one citation-ledger entry per constant (chapter/table/page **or** WITHHELD + the deposition needed), unit conventions, implied imperial constants, independent recomputation |
| `fem_vs_cole.csv` | per case: Cole levels, FEM wall peak, FEM impulse, the relative deviations, PASS/FAIL of every criterion |
| `grid_convergence.json` | three-level GCI per quantity per case, observed orders, energy-ledger residuals, criterion verdicts |
| `case_matrix.csv`, `srs_comparison.json` | chain-2 artifacts, extended to the full 6-case ensemble; `gci_*` recomputed with `roache_gci(fine, medium, coarse)` and non-demonstrated quantities marked `n/a` (see `srs_comparison.json` -> `extension`) |
| `chain_metrics.json` | chain-2 provenance record (mission 15095, cases C1-C4 as generated there; not rewritten) |

## 2. Anchor status: WITHHELD (criterion 1)

The constants are the repository's own declarations, not a transcription of Cole (1948): no copy of the monograph is reachable from the sandbox and outbound HTTP is blocked (Errno 99 / TLS timeout), so neither chapter/table number nor printed digits could be read.

| constant | value | unit | citation status |
|---|---|---|---|
| `K` | 5.240e+07 | Pa | WITHHELD |
| `A` | 1.130000 | 1 (dimensionless exponent) | WITHHELD |
| `K_theta` | 8.400e-05 | s | WITHHELD |
| `decay_exponent` | -0.240000 | 1 | WITHHELD |
| `K_b` | 2.110000 | s m^(5/6) kg^(-1/3) | WITHHELD |
| `bubble_exponent` | 0.833333 | 1 | DERIVED_NOT_NEEDED |

What a deposition must contain to close each entry is listed in `cole_anchor.json -> withheld_ledger.deposition_shopping_list`: the printed TNT peak-overpressure fit (K, A) with its unit system and validity range, the decay constant and its exponent, the bubble-period constant with the definition of H, and the charge-type / TNT-equivalence statement.

### Unit-convention checks that do not need the book

* bubble-period exponents re-derived (Rayleigh collapse) **and** re-recovered by regression: weight exponent 0.333333 (expected 1/3), head exponent -0.833333 (expected -5/6);
* the recorded metric constants imply K = 21604.9 psi and K_b = 4.363 in (lb, ft) form, so a deposited imperial-unit source can be checked against them in one step;
* the competing atmospheric-head conventions (10.33 m fresh, 10.08 m seawater, 33 ft = 10.058 m) move the bubble period by at most 0.45 %, i.e. they can neither rescue nor break a +/-15 % verdict.

### The AN-COLE catalog entry and a cross-implementation probe

| field | value |
|---|---|
| id | AN-COLE |
| klass | A |
| source | Cole R.H., Underwater Explosions, Princeton Univ. Press, 1948 |
| statement | p_m = K (W^(1/3)/R)^A, K and A being charge-dependent implementation parameters |

The catalog registers the FORM and the citation only -- no numeric table, so the catalog alone cannot close the numeric verification. It also publishes a numeric evaluation row for AN-COLE: at reduced range 0.23208 it reports p_m = 43.1869 MPa. Reproducing that row at A = 1.130 requires K = 2.250e+08 Pa, whereas this repository declares K = 5.240e+07 Pa -- a factor 4.29 apart at the *identical* reduced range (which is our own case C6, W=100 kg, R=20 m). The registry explicitly says its table values were not transcribed, so neither number can be adjudicated without the source: **CROSS_IMPLEMENTATION_CONFLICT_UNARBITRATED**.

### Independent recomputation

p_m, theta, T, specific impulse and shock energy were re-evaluated from formulas rewritten in `scripts/w24_cole_anchor_citation.py` for all 6 cases; max relative deviation against the recorded values = 0.000e+00. The exponents and coefficients were then *recovered* from the values by regression: A = 1.130000, K = 5.240e+07 Pa, n = -0.240000, K_theta = 8.400e-05 s. This proves transcription/arithmetic consistency -- and nothing about Cole's printed tables.

## 3. FEM vs Cole, case by case (criterion 2, +/-15 %)

| case | W [kg] | R [m] | W^(1/3)/R | Cole p_m [MPa] | FEM wall peak [MPa] | peak vs free field | peak vs 2 p_m | impulse window [us] | FEM impulse [Pa s] | Cole incident (same window) [Pa s] | dev vs Cole incident |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 5 | 12 | 0.1425 | 5.796 | 11.592 | 100.0 % (FAIL) | 0.0 % (PASS_TAUTOLOGICAL) | 99.4 | 389.96 | 467.31 | 16.6 % (FAIL) |
| C2 | 10 | 18 | 0.1197 | 4.759 | 9.519 | 100.0 % (FAIL) | 0.0 % (PASS_TAUTOLOGICAL) | 107.1 | 341.88 | 428.95 | 20.3 % (FAIL) |
| C3 | 25 | 25 | 0.1170 | 4.637 | 9.274 | 100.0 % (FAIL) | 0.0 % (PASS_TAUTOLOGICAL) | 114.2 | 354.27 | 462.30 | 23.4 % (FAIL) |
| C4 | 50 | 30 | 0.1228 | 4.899 | 9.799 | 100.0 % (FAIL) | 0.0 % (PASS_TAUTOLOGICAL) | 119.4 | 388.11 | 521.94 | 25.6 % (FAIL) |
| C5 | 2 | 25 | 0.0504 | 1.791 | 3.582 | 100.0 % (FAIL) | 0.0 % (PASS_TAUTOLOGICAL) | 98.8 | 118.67 | 142.06 | 16.5 % (FAIL) |
| C6 | 100 | 20 | 0.2321 | 10.058 | 20.115 | 100.0 % (FAIL) | 0.0 % (PASS_TAUTOLOGICAL) | 123.0 | 805.83 | 1109.48 | 27.4 % (FAIL) |

* Peak vs the Cole **free field** fails in 6/6 cases by exactly +100 %: the wetted face is loaded by `p_wall = 2 p_i - rho c v_n` and at t = 0 the panel has not moved, so the peak is *identically* 2 p_m. That is a property of the acoustic boundary condition, not of the solver, and no refinement can turn it into a PASS.
* Peak vs the **rigid-wall** 2 p_m passes at 0.0 % and is tautological for the same reason; it is tagged `PASS_TAUTOLOGICAL` so no reader mistakes it for validation.
* The comparison that can genuinely fail is the impulse over a common window (up to cavitation onset): 0/6 cases inside +/-15 %. The delivered impulse sits below the free-field Cole impulse because a flexible plate reflects less than a rigid wall and because the acoustic model is cut at cavitation -- that is physics, and the residual gap is what a cavitating-fluid model would have to recover.
* **Criterion validity.** The +/-15 % band is applied as instructed to every row, but the impulse row compares two different physical quantities (delivered wall impulse of a mass-loaded, cavitating panel vs free-field incident impulse): a FAIL there is a statement about the model class, not about solver accuracy. The solver-accuracy axis is the FEM-vs-Taylor row below.
* Solver verification is a separate axis: FEM vs the Taylor air-backed-plate closed form at the same incident wave gives interface-pressure / velocity errors of C1: 2.88 % / 4.44 %; C2: 2.91 % / 4.18 %; C3: 2.95 % / 3.96 %; C4: 2.92 % / 3.77 %; C5: 2.87 % / 4.48 %; C6: 2.96 % / 3.77 %.

### Rigid-limit coupling check (anchor bearing)

| case | FEM impulse, stiffened panel [Pa s] | 2 x Cole incident over [0, 4 theta] [Pa s] | deviation | verdict |
|---|---|---|---|---|
| C1 | 2593.81 | 2609.11 | 0.59 % | PASS |
| C2 | 2802.37 | 2814.62 | 0.44 % | PASS |
| C3 | 3730.77 | 3742.38 | 0.31 % | PASS |
| C4 | 4912.05 | 4924.09 | 0.24 % | PASS |
| C5 | 757.44 | 762.25 | 0.63 % | PASS |
| C6 | 10907.38 | 10931.90 | 0.22 % | PASS |

## 4. Grid convergence and energy (criteria 3 and 4)

Levels (nx, ny, steps/decay) = `[[12, 8, 40], [18, 12, 60], [27, 18, 90]]`, refinement ratio 1.5, Fs = 1.25. **GCI verdict: FAIL** -- 10 quantity/case pairs converged, 7 above 10 %, 7 not demonstrated. Not demonstrated: C1:peak_center_deflection_m, C1:wall_impulse_0_theta_pa_s, C3:wall_impulse_0_theta_pa_s, C4:wall_impulse_0_theta_pa_s, C5:peak_center_deflection_m, C5:wall_impulse_0_theta_pa_s, C6:wall_impulse_0_theta_pa_s.

Energy-ledger residual over the whole record: worst 0.006 (< 0.05) -> **PASS**.

Per-case GCI: values are listed coarse -> fine, the *observed* order p is reported unclipped (the GCI itself uses p clipped to [0.5, 4], Fs = 1.25):

| case | quantity | values (coarse -> fine) | p observed | p used | GCI_fine | status |
|---|---|---|---|---|---|---|
| C1 | peak_center_deflection_m | 0.0017007, 0.0016976, 0.0017078 | n/a | n/a | n/a | NOT_DEMONSTRATED |
| C1 | peak_center_acceleration_ms2 | 1.4604e+05, 1.4756e+05, 1.4767e+05 | 6.497 | 4.000 | 0.003 | CONVERGED_UNDER_LIMIT |
| C1 | wall_impulse_0_theta_pa_s | 155.94, 311.46, 271.15 | n/a | n/a | n/a | NOT_DEMONSTRATED |
| C1 | wall_impulse_0_4theta_pa_s | 187.78, 34.337, -7.5248 | 3.204 | 3.204 | 0.383 | ABOVE_LIMIT |
| C2 | peak_center_deflection_m | 0.0018656, 0.0018622, 0.0018589 | 0.141 | 0.500 | 0.010 | CONVERGED_UNDER_LIMIT |
| C2 | peak_center_acceleration_ms2 | 1.1992e+05, 1.2117e+05, 1.2126e+05 | 6.497 | 4.000 | 0.003 | CONVERGED_UNDER_LIMIT |
| C2 | wall_impulse_0_theta_pa_s | 134.23, 209.07, 227.79 | 3.419 | 3.419 | 0.232 | ABOVE_LIMIT |
| C2 | wall_impulse_0_4theta_pa_s | 202.67, 83.683, 25.269 | 1.755 | 1.755 | 0.708 | ABOVE_LIMIT |
| C3 | peak_center_deflection_m | 0.0025522, 0.0025366, 0.0025308 | 2.436 | 2.436 | 0.005 | CONVERGED_UNDER_LIMIT |
| C3 | peak_center_acceleration_ms2 | 1.1683e+05, 1.1805e+05, 1.1814e+05 | 6.497 | 4.000 | 0.003 | CONVERGED_UNDER_LIMIT |
| C3 | wall_impulse_0_theta_pa_s | 161.92, 143.8, 209.79 | n/a | n/a | n/a | NOT_DEMONSTRATED |
| C3 | wall_impulse_0_4theta_pa_s | 238.24, 150.08, 83.252 | 0.683 | 0.683 | 1.449 | ABOVE_LIMIT |
| C4 | peak_center_deflection_m | 0.0034542, 0.0034138, 0.0034027 | 3.181 | 3.181 | 0.006 | CONVERGED_UNDER_LIMIT |
| C4 | peak_center_acceleration_ms2 | 1.2344e+05, 1.2473e+05, 1.2482e+05 | 6.497 | 4.000 | 0.003 | CONVERGED_UNDER_LIMIT |
| C4 | wall_impulse_0_theta_pa_s | 215.08, 123.74, 194.59 | n/a | n/a | n/a | NOT_DEMONSTRATED |
| C4 | wall_impulse_0_4theta_pa_s | 268.74, 204.23, 138.14 | -0.060 | 0.500 | 1.335 | ABOVE_LIMIT |
| C5 | peak_center_deflection_m | 0.0004954, 0.0004944, 0.0005003 | n/a | n/a | n/a | NOT_DEMONSTRATED |
| C5 | peak_center_acceleration_ms2 | 45122, 45592, 45626 | 6.497 | 4.000 | 0.003 | CONVERGED_UNDER_LIMIT |
| C5 | wall_impulse_0_theta_pa_s | 48.675, 98.052, 83.315 | n/a | n/a | n/a | NOT_DEMONSTRATED |
| C5 | wall_impulse_0_4theta_pa_s | 54.125, 6.9442, -3.6158 | 3.692 | 3.692 | 0.314 | ABOVE_LIMIT |
| C6 | peak_center_deflection_m | 0.007759, 0.0076521, 0.0076221 | 3.131 | 3.131 | 0.007 | CONVERGED_UNDER_LIMIT |
| C6 | peak_center_acceleration_ms2 | 2.5342e+05, 2.5606e+05, 2.5625e+05 | 6.497 | 4.000 | 0.003 | CONVERGED_UNDER_LIMIT |
| C6 | wall_impulse_0_theta_pa_s | 484.04, 242.37, 377.56 | n/a | n/a | n/a | NOT_DEMONSTRATED |
| C6 | wall_impulse_0_4theta_pa_s | 558.17, 446.86, 318.71 | -0.347 | 0.500 | 1.109 | ABOVE_LIMIT |

Quantity-wise summary (converged / above limit / not demonstrated): peak_center_deflection_m: 4/0/2; peak_center_acceleration_ms2: 6/0/0; wall_impulse_0_theta_pa_s: 0/1/5; wall_impulse_0_4theta_pa_s: 0/6/0.

Two observables need a word of explanation, and neither is re-scored:

* the **peak centre acceleration** gives an observed order of 6.5 in every case (clipped to 4 for the GCI). The histories are identical in normalised time, so the same order appears case after case: the peak is a kink sampled at a mesh-dependent instant, which is not a smooth-functional convergence rate. Its GCI is small (0.3 %), so the practical conclusion is unaffected, but the order must not be quoted as a numerical-method property.
* the **net [0, 4 theta] impulse** is a cancellation, not a quantity: the loading part and the post-cavitation tensile part almost annihilate, the conditioning ratio |net| / (gross) is C1: 0.009, C2: 0.036, C3: 0.113, C4: 0.171, C5: 0.015, C6: 0.191. A GCI on such a number is meaningless, which is exactly what the 'above limit' statuses say. The physically admissible window is up to cavitation onset; its impulse is kept as a supplementary diagnostic (threshold-defined, so convergence is not claimed for it either).

## 5. Wording constraint (criterion 5)

Because the anchor is WITHHELD, no statement here or in the report claims agreement with Cole's formula. Defensible claims only: (i) the chain is internally consistent with the repository's transcription of the constants (self-check deviation 0.000e+00); (ii) the FEM solver is verified against the Taylor closed form; (iii) the wall peak is the rigid-wall doubling of whatever incident peak is applied, so it carries no independent information about Cole's digits.

## 6. Figures

* `docs/assets/w24_undex_cole/w24_fem_vs_cole.png`
* `docs/assets/w24_undex_cole/w24_grid_convergence.png`
* `docs/assets/w24_undex_cole/w24_energy_and_load.png`

## 7. Reproduction

```bash
PYTHONPATH=src python scripts/w24_cole_anchor_citation.py   # anchor + ledger
PYTHONPATH=src python scripts/w24_undex_fem_vs_cole.py      # FEM closure (~8 min)
PYTHONPATH=src pytest -q tests/test_undex_cole_chain3.py     # invariants
```

## 8. Limitations

* The fluid leg is analytic (local impedance, normal incidence); the structure is fully discretised. Oblique incidence, diffraction and bubble-pulse second loading are out of scope.
* The acoustic model admits tension after cavitation onset, so the pre-cavitation window is the only interval in which the wall pressure is physical.
* Constants are unverified (`WITHHELD`); every Cole number in the tables traces to `cole_anchor.json`, which traces to repository declarations. If a deposition arrives, re-running the two scripts regenerates every number here.
