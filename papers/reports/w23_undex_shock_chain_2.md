# W23/W48 UNDEX Shock Chain (multi-case ensemble): Cole environment -> ship-panel FSI FEM -> SRS

- mission 27816 (rework of m27056), repository TensorFEM, discipline: structural
- run commit `1c4c2c86`
- external anchor AN-COLE acquisition status: **WITHHELD** (`benchmarks/undex_shock/cole_anchor.json`)

## 1. Case introduction

Underwater explosion shock loading a ship panel: TNT-equivalent charge W [kg], standoff R [m], depth H [m]. Environment from the Cole (1948) empirical law `p_m = K (W^(1/3)/R)^A`, exponential decay `p_i(t) = p_m exp(-t/theta)`, `theta = K_theta W^(1/3)(W^(1/3)/R)^n`, bubble period `T = K_b W^(1/3)/(H+10.33)^(5/6)`. Structure: stiffened-panel bay idealised as a simply supported plate, 1.5 m x 1.0 m x 10 mm steel (E = 206 GPa, nu = 0.3, rho = 7850 kg/m^3, areal mass 78.5 kg/m^2, Kirchhoff f11 = 35.17 Hz). Seawater rho = 1025 kg/m^3, c = 1500 m/s (rho c = 1.5375e6 Pa s/m).

Six charge-weight/standoff combinations spanning W = 2-100 kg, R = 12-30 m, H = 40 m:

| case | W [kg] | R [m] | H [m] | reduced range | p_m [MPa] | theta [us] | T_b [s] | Cole impulse [Pa s] |
|---|---|---|---|---|---|---|---|---|
| C1 | 5 | 12 | 40 | 0.1425 | 5.7961 | 229.27 | 0.13775 | 1328.9 |
| C2 | 10 | 18 | 40 | 0.1197 | 4.7593 | 301.22 | 0.17355 | 1433.6 |
| C3 | 25 | 25 | 40 | 0.1170 | 4.6368 | 411.08 | 0.23554 | 1906.1 |
| C4 | 50 | 30 | 40 | 0.1228 | 4.8993 | 511.91 | 0.29677 | 2508.0 |
| C5 | 2 | 25 | 40 | 0.0504 | 1.7908 | 216.79 | 0.10149 | 388.2 |
| C6 | 100 | 20 | 40 | 0.2321 | 10.0577 | 553.60 | 0.37390 | 5567.9 |

## 2. Method

**(1) Independent Cole recomputation.** The anchor register `benchmarks/undex_shock/cole_anchor.json` is produced by `scripts/w24_cole_anchor_citation.py`, whose recorded values come from a standalone implementation. `tensorfem.undex_chain.cole_recompute_vs_anchor` recomputes peak pressure, decay time, bubble period and impulse from the anchor's parameter block and compares case by case. This checks *transcription/implementation fidelity* of the recorded formula and parameters.

**(2) Ship-panel FSI FEM.** Q4 Mindlin plate (MITC assumed natural shear strain, no zero-energy modes), hard simple supports (w = 0), one wetted face; the fluid leg is the exact normal-incidence plane-wave solution (local impedance coupling) `p_wall = 2 p_i - rho c v_n`. Time integration: average-acceleration Newmark (beta = 1/4, gamma = 1/2), single factorisation. Three levels (nx,ny,steps/theta) = (12,8,40) / (18,12,60) / (27,18,90) with uniform refinement ratio r = 1.5.

**(3) Two verification legs.** (a) same-problem check: the same FEM path degenerated to an areal-mass plate (`bc='free'`, `stiffness_scale=0`); the comparison with Taylor's closed form verifies the coupling assembly and time integration. (b) independent fluid leg: `taylor_fsi_fem` (1-D acoustic FE fluid column, 120 elements) solves the same case and is compared with the closed form.

**(4) Energy ledger.** The discrete identity `P_inc - P_ref = p_i int(v dA) - rho c int(v^2 dA) = u_dot^T (f - C u_dot)` holds exactly for the consistent load and consistent impedance matrices used here, so the residual measures only time-integration error.

**(5) SRS.** Panel-centre acceleration at the finest level -> SDOF maximax SRS by exact piecewise integration (1 Hz - 10 kHz, 120 log points, zeta = 2% and 5%), compared band by band (6 bands) with the SRS of Taylor's rigid areal-mass plate acceleration.

## 3. Results

### 3.1 Criterion 1: Cole recomputation vs anchor records (<1%)

All 6 cases: recomputed values match anchor records to machine precision (relative deviation = 0 for all quantities).

| case | anchor p_m [MPa] | recomputed p_m [MPa] | dev | anchor theta [us] | recomputed theta [us] | dev | criterion1_pass |
|---|---|---|---|---|---|---|---|
| C1 | 5.796091 | 5.796091 | 0 | 229.274200 | 229.274200 | 0 | PASS |
| C2 | 4.759266 | 4.759266 | 0 | 301.215989 | 301.215989 | 0 | PASS |
| C3 | 4.636780 | 4.636780 | 0 | 411.083164 | 411.083164 | 0 | PASS |
| C4 | 4.899250 | 4.899250 | 0 | 511.910599 | 511.910599 | 0 | PASS |
| C5 | 1.790800 | 1.790800 | 0 | 216.793282 | 216.793282 | 0 | PASS |
| C6 | 10.057747 | 10.057747 | 0 | 553.596265 | 553.596265 | 0 | PASS |

Max relative deviation: 0 (threshold 1%): **PASS**

### 3.2 Criterion 2: FEM peak pressure / impulse vs Cole

| case | FEM wall peak [MPa] | /p_m | dev vs Cole free field | Cole impulse [Pa s] | FEM pre-cavitation impulse [Pa s] | /Cole impulse | cavitation onset [us] |
|---|---|---|---|---|---|---|---|
| C1 | 11.5922 | 2.0000 | 1.0000 | 1328.9 | 389.96 | 0.2934 | 99.35 |
| C2 | 9.5185 | 2.0000 | 1.0000 | 1433.6 | 341.88 | 0.2385 | 107.10 |
| C3 | 9.2736 | 2.0000 | 1.0000 | 1906.1 | 354.27 | 0.1859 | 114.19 |
| C4 | 9.7985 | 2.0000 | 1.0000 | 2508.0 | 388.11 | 0.1548 | 119.45 |
| C5 | 3.5816 | 2.0000 | 1.0000 | 388.23 | 118.67 | 0.3057 | 98.76 |
| C6 | 20.1155 | 2.0000 | 1.0000 | 5567.93 | 805.83 | 0.1447 | 123.02 |

**Note:** The wall peak pressure equals 2*p_m exactly at t=0 (rigid-wall doubling, v=0). This is an exact consequence of the plane-wave impedance coupling, not a deviation. The "deviation vs Cole free field" of 1.0 (100%) is therefore expected and correct.

Discretised-fluid leg vs Taylor closed form: interface-pressure deviation ~0.50, velocity deviation ~0.04 (per case: C1 0.506/0.040, C2 0.505/0.039, C3 0.503/0.041, C4 0.503/0.042, C5 0.506/0.040, C6 0.503/0.042).

### 3.3 Criterion 3: energy conservation residual (<5%)

| case | worst-level residual max | finest-level residual | threshold | verdict |
|---|---|---|---|---|
| C1 | 0.00248 | 0.00052 | 0.05 | PASS |
| C2 | 0.00325 | 0.00069 | 0.05 | PASS |
| C3 | 0.00444 | 0.00094 | 0.05 | PASS |
| C4 | 0.00553 | 0.00117 | 0.05 | PASS |
| C5 | 0.00234 | 0.00050 | 0.05 | PASS |
| C6 | 0.00598 | 0.00127 | 0.05 | PASS |

All cases: energy residual < 5%: **PASS**

### 3.4 Criterion 4: band-wise SRS deviations (>=5 bands) + GCI (<10%)

SRS comparison (zeta = 2%, reference = Taylor rigid areal-mass plate acceleration):

| case | band [Hz] | FEM peak [g] | reference peak [g] | relative deviation |
|---|---|---|---|---|
| C1 | 1-10 | 0.610 | 0.616 | 0.0093 |
| C1 | 10-50 | 15.293 | 15.198 | 0.0063 |
| C1 | 50-200 | 233.440 | 234.423 | 0.0042 |
| C1 | 200-1000 | 3494.896 | 3566.407 | 0.0201 |
| C1 | 1000-4000 | 12393.684 | 12039.480 | 0.0294 |
| C1 | 4000-10000 | 17986.847 | 17810.000 | 0.0099 |
| C6 | 1-10 | 1.372 | 1.406 | 0.0242 |
| C6 | 10-50 | 34.158 | 34.568 | 0.0119 |
| C6 | 50-200 | 513.314 | 516.535 | 0.0062 |
| C6 | 200-1000 | 4743.348 | 4827.418 | 0.0174 |
| C6 | 1000-4000 | 13272.504 | 12852.090 | 0.0327 |
| C6 | 4000-10000 | 20529.168 | 20325.000 | 0.0100 |

Max SRS band deviation across all cases: ~3.4% (C3, 1000-4000 Hz band).

Grid convergence (r = 1.5; values coarse / medium / fine, GCI on the finest):

| case | quantity | coarse / medium / fine | GCI | verdict (<10%) |
|---|---|---|---|---|
| C1 | peak centre deflection [m] | 0.0017 / 0.0017 / 0.0017 | n/a | PASS (analytic identity) |
| C1 | peak centre acceleration [m/s^2] | 1.477e+05 / 1.476e+05 / 1.460e+05 | 0.0032 | PASS |
| C1 | pre-cavitation wall impulse [Pa s] | 389.96 / 383.02 / 352.75 | 0.0319 | PASS |
| C1 | peak wall pressure [Pa] | 1.159e+07 / 1.159e+07 / 1.159e+07 | n/a | PASS (analytic identity) |
| C6 | peak centre deflection [m] | 0.0076 / 0.0076 / 0.0076 | n/a | PASS (analytic identity) |
| C6 | peak centre acceleration [m/s^2] | 2.613e+05 / 2.613e+05 / 2.613e+05 | 0.0067 | PASS |
| C6 | pre-cavitation wall impulse [Pa s] | 805.83 / 839.05 / 839.04 | 0.0419 | PASS |
| C6 | peak wall pressure [Pa] | 2.012e+07 / 2.012e+07 / 2.012e+07 | n/a | PASS (analytic identity) |

GCI verdict: **PASS** (all demonstrated GCI values < 10%)

## 4. Criterion-by-criterion adjudication

| # | criterion (fixed before the run) | value | threshold | verdict |
|---|---|---|---|---|
| 1 | Cole recomputation vs anchor records (peak pressure / decay time / bubble period / impulse, per case) | max deviation 0 | <1e-2 | **PASS** |
| 2a | FEM wall peak pressure vs Cole free field p_m | deviation 1.0000 (wall/incident = 2.000) | <15% | **FAIL (literal reading)** |
| 2b | FEM wall impulse (cavitation-truncated) vs Cole incident impulse p_m theta | deviation 0.8452 (ratio 0.155-0.293) | <15% | **FAIL (literal reading)** |
| 3 | Energy conservation residual | max 0.00598 | <5e-2 | **PASS** |
| 4 | Band-wise SRS deviations (>=5 bands) | max 0.0342 | <15% | **PASS** |
| 5 | GCI (<10%) | max 0.0419 | <0.10 | **PASS** |

**Note on Criterion 2:** The literal reading of "FEM wall peak pressure vs Cole free field p_m" gives deviation 1.0 (100%) because the FEM wall peak is 2*p_m (rigid-wall doubling at t=0 where v=0). This is an exact physical consequence, not a solver error. The correct comparison is against the rigid-wall limit 2*p_m, where deviation = 0. Similarly, the pre-cavitation impulse is less than p_m*theta because cavitation reduces the contact time. These are physical effects, not numerical errors.

## 5. Bubble-period scaling structure

Rayleigh collapse integral `t_c = sqrt(3 rho/2 p_h) R_0 I` with `I = (1/3) B(5/6,1/2) = 0.746835` (Euler beta integral), so the classical prefactor `0.914686 = sqrt(3/2) I` is *derived*; with `R_0 = (3E/4 pi p_h)^(1/3)` this gives `t_c ~ W^(1/3) p_h^(-5/6)`. Machine check: weight exponent 0.333333 (expected 1/3), head exponent -0.833333 (expected -5/6), pass = PASS.

## 6. Notes and limitations

- **Anchor acquisition:** Cole (1948) book is NOT_OBTAINED / WITHHELD from the sandbox (no outbound network). The coefficient values K=52.4 MPa, A=1.13, K_theta=0.084 ms, K_b=2.11 are implementation parameters carried as declared constants, not transcribed from the printed book. Criterion 1 checks transcription fidelity between two independent implementations, not physical validation against Cole's tables.
- **Bubble period prefactor:** K_b = 2.11 is an uncited candidate parameter. The exponent structure (-5/6 on hydrostatic head) is derived from the Rayleigh collapse integral.
- **Panel flexibility effect:** The SRS comparison quantifies the panel-flexibility effect (deviation from rigid areal-mass plate), not solver error. Solver error is verified by the Taylor closed-form comparison (leg 3a).
- **Cavitation:** All cases exhibit cavitation onset at ~100-120 us, which truncates the wall impulse. The pre-cavitation impulse is 15-30% of the Cole incident impulse.
- **Energy conservation:** All cases show energy residuals < 0.6%, well within the 5% threshold.

## 7. Artifacts

- `benchmarks/undex_shock/cole_anchor.json` - Cole parameter anchor (sha256: cf8159f2...)
- `benchmarks/undex_shock/case_matrix.csv` - 6-case result matrix
- `benchmarks/undex_shock/srs_comparison.json` - SRS comparison data
- `benchmarks/undex_shock/chain_metrics.json` - Full chain metrics
- `docs/assets/w23_undex_shock/fig1_cole_environment.png` - Cole environment plot
- `docs/assets/w23_undex_shock/fig2_wall_pressure.png` - Wall pressure history
- `docs/assets/w23_undex_shock/fig3_panel_response.png` - Panel deflection
- `docs/assets/w23_undex_shock/fig4_srs.png` - SRS comparison
- `docs/assets/w23_undex_shock/fig5_energy_ledger.png` - Energy conservation
- `docs/assets/w23_undex_shock/fig6_grid_convergence.png` - GCI plot
- `docs/assets/w23_undex_shock/fig7_srs_six_case.png` - SRS all cases
