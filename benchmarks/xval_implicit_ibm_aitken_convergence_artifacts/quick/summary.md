# XVAL convergence study — implicit IBM + Aitken added mass

Generated: 2026-11-15T16:15:42Z  |  wall 56 s  |  acceptance 15 %

Reference: Lamb 1932 art. 71, `m_a = rho_f pi c^2` per unit length; coupled frequency `f/f_vac = sqrt(m_s/(m_s+m_a))`.  Coupling: implicit direct forcing IBM with Aitken `Delta^2` sub-iterations (`tensorfem.ibm_aitken_added_mass`).

## Measured added mass vs analytic

| case | c [dx] | L [dx] | 2c/L | ds [dx] | omega | ratio m_a/rho pi c^2 | f/f_vac meas | f/f_vac analytic | err % | <15 % |
|---|---|---|---|---|---|---|---|---|---|---|
| quick_c6_L120 | 6 | 120 | 0.100 | 1.00 | 1.90 | 2.2079 | 0.6559 | 0.7906 | 17.04 | NO |
| quick_c6_L36 | 6 | 36 | 0.333 | 1.00 | 1.90 | 2.0313 | 0.6713 | 0.7906 | 15.08 | NO |
| quick_c6_L72 | 6 | 72 | 0.167 | 1.00 | 1.90 | 2.1689 | 0.6592 | 0.7906 | 16.62 | NO |
| quick_c6_L27 | 6 | 27 | 0.444 | 1.00 | 1.90 | 1.9000 | 0.6836 | 0.7906 | 13.53 | yes |

## Fits and extrapolation

- confinement axis (`c = 16 dx`): `r = 2.2403 + -1.1472 b^1.55`, rms 0.0000 -> `b -> 0` limit **2.2403**
- one-parameter smearing model `r = (1 + delta/c)^2`: **delta = 2.65 dx** (rms 5.85 % of the mean ratio, max residual 0.178, n = 4)

### Resolution required to reach a given coupled-frequency error

| target `|f/f_vac - analytic|` | allowed `m_a/m_a,Lamb` | required `c/delta` | required `c` [dx] |
|---|---|---|---|
| 3 % | 1.167 | 12.4 | 33 |
| 15 % | 2.024 | 2.4 | 6 |

## Figures

- `xval_conv_records.png`
- `xval_conv_confinement.png`
- `xval_conv_resolution.png`
- `xval_conv_verdict.png`
- `xval_conv_inflation_collapse.png`

## Limitations (honest)

- `alpha` and `beta` are **empirical** fit coefficients: no closed-form added mass exists for the discrete smeared IBM body, so the model is a quantified bias decomposition, not a derivation.
- Two-axis extrapolation from a finite grid: the limit inherits the fit residual quoted above; the largest simulated configuration (`c` up to 6 dx) is still far from `dx/c -> 0` in an absolute sense.
- Periodic box, quiescent fluid: no inflow/walls, so this validates the coupling algorithm and the inertial (potential) added mass only.
- Degenerate/oscillatory estimator agreement is reported per case (`period_zc`, `period_damped`, `period_fft`); any case whose damped fit is invalid falls back to zero crossings and is flagged.
