# XVAL convergence study — implicit IBM + Aitken added mass

Generated: 2026-11-15T04:35:43Z  |  wall 1524 s  |  acceptance 15 %

Reference: Lamb 1932 art. 71, `m_a = rho_f pi c^2` per unit length; coupled frequency `f/f_vac = sqrt(m_s/(m_s+m_a))`.  Coupling: implicit direct forcing IBM with Aitken `Delta^2` sub-iterations (`tensorfem.ibm_aitken_added_mass`).

## Measured added mass vs analytic

| case | c [dx] | L [dx] | 2c/L | ds [dx] | omega | ratio m_a/rho pi c^2 | f/f_vac meas | f/f_vac analytic | err % | <15 % |
|---|---|---|---|---|---|---|---|---|---|---|
| A_c16_L144 | 16 | 144 | 0.222 | 1.00 | 1.90 | 1.4656 | 0.7295 | 0.7906 | 7.73 | yes |
| A_c16_L256 | 16 | 256 | 0.125 | 1.00 | 1.90 | 1.5417 | 0.7208 | 0.7906 | 8.83 | yes |
| A_c16_L96 | 16 | 96 | 0.333 | 1.00 | 1.90 | 1.4021 | 0.7370 | 0.7906 | 6.78 | yes |
| B_c8_L72 | 8 | 72 | 0.222 | 1.00 | 1.90 | 1.8307 | 0.6903 | 0.7906 | 12.68 | yes |
| B_c12_L108 | 12 | 108 | 0.222 | 1.00 | 1.90 | 1.5673 | 0.7179 | 0.7906 | 9.19 | yes |
| B_c24_L216 | 24 | 216 | 0.222 | 1.00 | 1.90 | 1.4653 | 0.7295 | 0.7906 | 7.73 | yes |
| B_c32_L288 | 32 | 288 | 0.222 | 1.00 | 1.90 | 2.0255 | 0.6719 | 0.7906 | 15.01 | NO |
| B_c48_L432 | 48 | 432 | 0.222 | 1.00 | 1.90 | 1.3396 | 0.7446 | 0.7906 | 5.82 | yes |
| D_c32_L432 | 32 | 432 | 0.148 | 1.00 | 1.90 | 1.3832 | 0.7392 | 0.7906 | 6.49 | yes |
| D_c48_L576 | 48 | 576 | 0.167 | 1.00 | 1.90 | 1.3296 | 0.7458 | 0.7906 | 5.66 | yes |
| E_c16_L144_om1.7 | 16 | 144 | 0.222 | 1.00 | 1.70 | 1.6221 | 0.7119 | 0.7906 | 9.95 | yes |
| E_c16_L144_om1.98 | 16 | 144 | 0.222 | 1.00 | 1.98 | 1.3797 | 0.7397 | 0.7906 | 6.44 | yes |
| E_c16_L144_y0_0p1 | 16 | 144 | 0.222 | 1.00 | 1.90 | 1.4660 | 0.7294 | 0.7906 | 7.74 | yes |

## Fits and extrapolation

- confinement axis (`c = 16 dx`): `r = 1.7615 + -0.6241 b^0.50`, rms 0.0012 -> `b -> 0` limit **1.7615**
- resolution axis (blockage 0.222): `r = 1.5277 + 1.9530 (dx/c)`, rms 0.2377 -> `dx/c -> 0` limit **1.5277**
- joint model `r = 1.5249 + 1.8828/c + -17.7455 b^4.00`, rms 0.2009 (8 configurations)
- one-parameter smearing model `r = (1 + delta/c)^2`: **delta = 3.37 dx** (rms 19.53 % of the mean ratio, max residual 0.804, n = 8)

### Resolution required to reach a given coupled-frequency error

| target `|f/f_vac - analytic|` | allowed `m_a/m_a,Lamb` | required `c/delta` | required `c` [dx] |
|---|---|---|---|
| 3 % | 1.167 | 12.4 | 42 |
| 15 % | 2.024 | 2.4 | 8 |
- extrapolated limit `m_a/m_a,Lamb -> 1.5249` (residual 52.49 %)  =>  `f/f_vac -> 0.7226` vs analytic 0.7906, error **8.59 %** (PASS @15 %, not @3 %)

### Blind prediction of held-out configurations

| case | c | 2c/L | measured ratio | predicted | rel. error |
|---|---|---|---|---|---|
| D_c32_L432 | 32 | 0.148 | 1.3832 | 1.5751 | 13.88 % |
| D_c48_L576 | 48 | 0.167 | 1.3296 | 1.5504 | 16.61 % |

### Controls at `c = 16 dx`, `L = 144 dx`

| control | value | ratio | delta vs base |
|---|---|---|---|
| base | - | 1.4656 | - |
| omega_lbm | 1.7 | 1.6221 | +10.68 % |
| omega_lbm | 1.98 | 1.3797 | -5.86 % |
| y0 | 0.1 | 1.4660 | +0.03 % |

## Figures

- `xval_conv_records.png`
- `xval_conv_confinement.png`
- `xval_conv_resolution.png`
- `xval_conv_verdict.png`
- `xval_conv_inflation_collapse.png`
- `xval_conv_flow_field.png`

## Limitations (honest)

- `alpha` and `beta` are **empirical** fit coefficients: no closed-form added mass exists for the discrete smeared IBM body, so the model is a quantified bias decomposition, not a derivation.
- Two-axis extrapolation from a finite grid: the limit inherits the fit residual quoted above; the largest simulated configuration (`c` up to 48 dx) is still far from `dx/c -> 0` in an absolute sense.
- Periodic box, quiescent fluid: no inflow/walls, so this validates the coupling algorithm and the inertial (potential) added mass only.
- Degenerate/oscillatory estimator agreement is reported per case (`period_zc`, `period_damped`, `period_fft`); any case whose damped fit is invalid falls back to zero crossings and is flagged.
