# W23 UNDEX shock chain (multi-case ensemble): Cole environment -> ship-panel FSI FEM -> SRS

- mission 15095, repository TensorFEM, discipline: structural
- run commit `fe26e5fe67162caaab36f7a46cd535a771e65ab7`
- external anchor AN-COLE acquisition status: **NOT_OBTAINED** (`benchmarks/undex_shock/cole_anchor.json`, sha256 `cf8159f22306a6b9...`)

## 1. Case introduction

Underwater explosion shock loading a ship panel: TNT-equivalent charge W [kg], standoff R [m], depth H [m].  Environment from the Cole (1948) empirical law `p_m = K (W^(1/3)/R)^A`, exponential decay `p_i(t) = p_m exp(-t/theta)`, `theta = K_theta W^(1/3)(W^(1/3)/R)^n`, bubble period `T = K_b W^(1/3)/(H+10.33)^(5/6)`.  Structure: stiffened-panel bay idealised as a simply supported plate, 1.5 m x 1.0 m x 10 mm steel (E = 206 GPa, nu = 0.3, rho = 7850 kg/m^3, areal mass 78.5 kg/m^2, Kirchhoff f11 = 35.17 Hz).  Seawater rho = 1025 kg/m^3, c = 1500 m/s (rho c = 1.5375e6 Pa s/m).

| case | W [kg] | R [m] | H [m] | reduced range | p_m [MPa] | theta [us] | T_b [s] | Cole impulse [Pa s] |
|---|---|---|---|---|---|---|---|---|
| C1 | 5 | 12 | 40 | 0.1425 | 5.7961 | 229.27 | 0.13775 | 1328.9 |
| C2 | 10 | 18 | 40 | 0.1197 | 4.7593 | 301.22 | 0.17355 | 1433.6 |
| C3 | 25 | 25 | 40 | 0.1170 | 4.6368 | 411.08 | 0.23554 | 1906.1 |
| C4 | 50 | 30 | 40 | 0.1228 | 4.8993 | 511.91 | 0.29677 | 2508.0 |

## 2. Method

**(1) Independent Cole recomputation.** The anchor register `benchmarks/undex_shock/cole_anchor.json` is produced by `scripts/w23_cole_anchor_record.py`, whose recorded values come from a standalone implementation that does not import tensorfem (power form and exp/log form, cross-checked).  `tensorfem.undex_chain.cole_recompute_vs_anchor` recomputes peak pressure, decay time, bubble period and impulse from the anchor's parameter block and compares case by case.  This checks *transcription/implementation fidelity* of the recorded formula and parameters -- it is **not** a physical validation against the printed Cole (1948) values, which is BLOCKED (see section 5).

**(2) Ship-panel FSI FEM.** Q4 Mindlin plate (MITC assumed natural shear strain, no zero-energy modes), hard simple supports (w = 0), one wetted face; the fluid leg is the exact normal-incidence plane-wave solution (local impedance coupling) `p_wall = 2 p_i - rho c v_n`, where `2 p_i` is the rigid-wall doubling and `- rho c v_n` the consistent impedance matrix -- both discretised with the same bilinear shape functions as the structure.  Time integration: average-acceleration Newmark (beta = 1/4, gamma = 1/2), single factorisation.  Three levels (nx,ny,steps/theta) = (12,8,40) / (18,12,60) / (27,18,90) with uniform refinement ratio r = 1.5.

**(3) Two verification legs.** (a) same-problem check: the same FEM path degenerated to an areal-mass plate (`bc='free'`, `stiffness_scale=0`, no bending, no supports); the consistent mass, impedance and load matrices then make rigid translation an exact solution of the discrete system, so the comparison with Taylor's closed form verifies the coupling assembly and the time integration (`tests/test_undex_chain.py::test_areal_mass_verification`). (b) independent fluid leg: the base module `taylor_fsi_fem` (1-D acoustic FE fluid column, 120 elements, plus areal-mass plate and exact outgoing truncation) solves the same case and is compared with the closed form.

**(4) Energy ledger.** The discrete identity `P_inc - P_ref = p_i int(v dA) - rho c int(v^2 dA) = u_dot^T (f - C u_dot)` holds exactly for the consistent load and consistent impedance matrices used here, so the residual measures only time-integration error: `E_inc = int p_i^2/(rho c) dA dt`, `E_ref = int (p_i - rho c v_h)^2/(rho c) dA dt`, `E_plate = 1/2 u_dot^T M u_dot + 1/2 u^T K u`.

**(5) SRS.** Panel-centre acceleration at the finest level -> SDOF maximax SRS by exact piecewise integration (1 Hz - 10 kHz, 120 log points, zeta = 2%% and 5%%), compared band by band (6 bands) with the SRS of Taylor's rigid areal-mass plate acceleration.  That comparison quantifies the **panel-flexibility effect**, not solver error (solver error is leg (3)).

## 3. Results

### 3.1 Criterion 1: Cole recomputation vs anchor records (<1%%)

| case | anchor p_m [MPa] | recomputed p_m [MPa] | dev | anchor theta [us] | recomputed theta [us] | dev | anchor T_b [s] | recomputed T_b [s] | dev | impulse dev |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 5.796091 | 5.796091 | 0 | 229.274200 | 229.274200 | 0 | 0.13774738 | 0.13774738 | 0 | 0 |
| C2 | 4.759266 | 4.759266 | 0 | 301.215989 | 301.215989 | 0 | 0.17355082 | 0.17355082 | 0 | 0 |
| C3 | 4.636780 | 4.636780 | 0 | 411.083164 | 411.083164 | 0 | 0.23554471 | 0.23554471 | 0 | 0 |
| C4 | 4.899250 | 4.899250 | 0 | 511.910599 | 511.910599 | 0 | 0.29676773 | 0.29676773 | 0 | 0 |

Max relative deviation 0 (threshold 1%): floating-point level, i.e. the two independent implementations agree; this does not validate the coefficient values K, A, K_theta, K_b themselves.

**Coefficient-free bubble-period scaling.** Rayleigh collapse integral `t_c = sqrt(3 rho/2 p_h) R_0 I` with `I = (1/3) B(5/6,1/2) = 0.746835` (Euler beta integral), so the classical prefactor `0.914686 = sqrt(3/2) I` is *derived*; with `R_0 = (3E/4 pi p_h)^(1/3)` this gives `t_c ~ W^(1/3) p_h^(-5/6)`.  Machine check: weight exponent 0.333333 (expected 1/3), head exponent -0.833333 (expected -5/6), pass = PASS.  So the exponent structure of the bubble law is derivable; only the prefactor K_b = 2.1100 is empirical and carried as an untranscribed candidate.

### 3.2 Criterion 2: FEM peak pressure / impulse vs Cole, per case

Four reference levels must be kept apart: Cole free field `p_m`, `p_m theta`; the rigid-wall limit `2 p_m`, `2 p_m theta` (reflection doubling); and the Taylor closed form (the reflection reduction a movable plate actually gives). The FEM wetted-face peak occurs at t = 0 (where v = 0) and therefore equals `2 p_m` exactly -- an exact consequence of reflection doubling.

| case | FEM wall peak [MPa] | /p_m | dev vs Cole free field | Cole impulse [Pa s] | FEM pre-cavitation impulse [Pa s] | /Cole impulse | cavitation onset [us] | wall tension [MPa] |
|---|---|---|---|---|---|---|---|---|
| C1 | 11.5922 | 2.0000 | 1.0000 | 1328.9 | 390.0 | 0.2934 | 99.4 | -1.1242 |
| C2 | 9.5185 | 2.0000 | 1.0000 | 1433.6 | 341.9 | 0.2385 | 107.1 | -0.7605 |
| C3 | 9.2736 | 2.0000 | 1.0000 | 1906.1 | 354.3 | 0.1859 | 114.2 | -0.6699 |
| C4 | 9.7985 | 2.0000 | 1.0000 | 2508.0 | 388.1 | 0.1548 | 119.4 | -0.6811 |

Discretised-fluid leg vs Taylor closed form: interface-pressure deviation 0.5060, velocity deviation 0.0421 (per case pressure/velocity: C1 0.5060/0.0397, C2 0.5046/0.0388, C3 0.5034/0.0410, C4 0.5027/0.0421); its pre-cavitation impulse deviates from the Cole incident impulse by C1 0.7006, C2 0.7553, C3 0.8080, C4 0.8393.

Panel-flexibility effect (relative to the rigid areal-mass plate): deflection ratio 0.9879-1.0430, centre-velocity deviation 0.0377-0.0444, wall-pressure deviation 0.0288-0.0295.

### 3.3 Criterion 3: energy conservation residual (<5%%)

| case | worst-level residual max | finest-level residual | threshold |
|---|---|---|---|
| C1 | 0.0025 | 5.247e-04 | 5e-2 |
| C2 | 0.0033 | 6.894e-04 | 5e-2 |
| C3 | 0.0044 | 9.408e-04 | 5e-2 |
| C4 | 0.0055 | 0.0012 | 5e-2 |

### 3.4 Criterion 4: band-wise SRS deviations (>=5 bands) + GCI (<10%%)

SRS comparison (zeta = 2%%, reference = Taylor rigid areal-mass plate acceleration):

| case | band [Hz] | FEM peak [g] | reference peak [g] | relative deviation | FEM peak frequency [Hz] |
|---|---|---|---|---|---|
| C1 | 1-10 | 0.610 | 0.616 | 0.0093 | 9.4 |
| C1 | 10-50 | 15.293 | 15.198 | 0.0063 | 47.9 |
| C1 | 50-200 | 233.440 | 234.423 | 0.0042 | 193.1 |
| C1 | 200-1000 | 3494.896 | 3566.407 | 0.0201 | 980.8 |
| C1 | 1000-4000 | 12393.684 | 12039.480 | 0.0294 | 3950.4 |
| C1 | 4000-10000 | 17986.847 | 17810.000 | 0.0099 | 10000.0 |
| C2 | 1-10 | 0.649 | 0.663 | 0.0222 | 9.4 |
| C2 | 10-50 | 16.067 | 16.096 | 0.0018 | 47.9 |
| C2 | 50-200 | 245.375 | 246.526 | 0.0047 | 193.1 |
| C2 | 200-1000 | 3147.309 | 3207.342 | 0.0187 | 980.8 |
| C2 | 1000-4000 | 10118.727 | 9807.941 | 0.0317 | 3950.4 |
| C2 | 4000-10000 | 15082.030 | 14890.788 | 0.0128 | 10000.0 |
| C3 | 1-10 | 0.850 | 0.879 | 0.0337 | 9.4 |
| C3 | 10-50 | 20.563 | 20.802 | 0.0114 | 47.9 |
| C3 | 50-200 | 310.629 | 312.394 | 0.0057 | 193.1 |
| C3 | 200-1000 | 3291.433 | 3350.786 | 0.0177 | 980.8 |
| C3 | 1000-4000 | 9726.372 | 9404.854 | 0.0342 | 3950.4 |
| C3 | 4000-10000 | 14943.284 | 14815.077 | 0.0087 | 10000.0 |
| C4 | 1-10 | 1.115 | 1.154 | 0.0338 | 9.4 |
| C4 | 10-50 | 26.194 | 26.669 | 0.0178 | 47.9 |
| C4 | 50-200 | 388.068 | 390.683 | 0.0067 | 193.1 |
| C4 | 200-1000 | 3588.904 | 3652.228 | 0.0173 | 980.8 |
| C4 | 1000-4000 | 10164.549 | 9856.031 | 0.0313 | 3950.4 |
| C4 | 4000-10000 | 15817.962 | 15708.186 | 0.0070 | 10000.0 |

Grid convergence (r = 1.5; values coarse / medium / fine, GCI on the finest):

| case | quantity | coarse / medium / fine | observed order p | GCI | verdict (<10%%) |
|---|---|---|---|---|---|
| C1 | peak centre deflection [m] | 0.0017 / 0.0017 / 0.0017 | n/a | 0 (identical at all levels) | PASS (analytic identity: v = 0 at t = 0 => max|p_wall| = 2 p_m) |
| C1 | peak centre acceleration [m/s^2] | 1.477e+05 / 1.476e+05 / 1.460e+05 | -6.497 | 0.0041 | PASS |
| C1 | pre-cavitation wall impulse [Pa s] | 389.9557 / 383.0153 / 352.7464 | -3.632 | 0.0990 | PASS |
| C1 | peak wall pressure [Pa] | 1.159e+07 / 1.159e+07 / 1.159e+07 | n/a | 0 (identical at all levels) | PASS (analytic identity: v = 0 at t = 0 => max|p_wall| = 2 p_m) |
| C2 | peak centre deflection [m] | 0.0019 / 0.0019 / 0.0019 | -0.141 | 0.0098 | PASS |
| C2 | peak centre acceleration [m/s^2] | 1.213e+05 / 1.212e+05 / 1.199e+05 | -6.497 | 0.0041 | PASS |
| C2 | pre-cavitation wall impulse [Pa s] | 341.8809 / 343.6847 / 305.5311 | n/a | 0 (identical at all levels) | PASS (analytic identity: v = 0 at t = 0 => max|p_wall| = 2 p_m) |
| C2 | peak wall pressure [Pa] | 9.519e+06 / 9.519e+06 / 9.519e+06 | n/a | 0 (identical at all levels) | PASS (analytic identity: v = 0 at t = 0 => max|p_wall| = 2 p_m) |
| C3 | peak centre deflection [m] | 0.0025 / 0.0025 / 0.0026 | -2.436 | 0.0128 | PASS |
| C3 | peak centre acceleration [m/s^2] | 1.181e+05 / 1.180e+05 / 1.168e+05 | -6.497 | 0.0041 | PASS |
| C3 | pre-cavitation wall impulse [Pa s] | 354.2694 / 367.3008 / 311.8858 | n/a | 0 (identical at all levels) | PASS (analytic identity: v = 0 at t = 0 => max|p_wall| = 2 p_m) |
| C3 | peak wall pressure [Pa] | 9.274e+06 / 9.274e+06 / 9.274e+06 | n/a | 0 (identical at all levels) | PASS (analytic identity: v = 0 at t = 0 => max|p_wall| = 2 p_m) |
| C4 | peak centre deflection [m] | 0.0034 / 0.0034 / 0.0035 | -3.181 | 0.0182 | PASS |
| C4 | peak centre acceleration [m/s^2] | 1.248e+05 / 1.247e+05 / 1.234e+05 | -6.497 | 0.0041 | PASS |
| C4 | pre-cavitation wall impulse [Pa s] | 388.1126 / 410.1432 / 340.1486 | n/a | 0 (identical at all levels) | PASS (analytic identity: v = 0 at t = 0 => max|p_wall| = 2 p_m) |
| C4 | peak wall pressure [Pa] | 9.799e+06 / 9.799e+06 / 9.799e+06 | n/a | 0 (identical at all levels) | PASS (analytic identity: v = 0 at t = 0 => max|p_wall| = 2 p_m) |

## 4. Criterion-by-criterion adjudication

| # | criterion (fixed before the run) | value | threshold | verdict | evidence pointer |
|---|---|---|---|---|---|
| 1 | Cole recomputation vs anchor records (peak pressure / decay time / bubble period / impulse, per case) | max deviation 0 | <1e-2 | PASS | `benchmarks/undex_shock/cole_anchor.json`, `chain_metrics.json:criterion1_recompute` |
| 2a | FEM wall peak pressure vs Cole free field p_m | deviation 1.0000 (wall/incident = 2.000) | <15% | **FAIL (literal reading)** | `case_matrix.csv:wall_peak_rel_dev_vs_cole` |
| 2b | FEM wall impulse (cavitation-truncated) vs Cole incident impulse p_m theta | deviation 0.8452 (ratio 0.155-0.293) | <15% | **FAIL (literal reading)** | `case_matrix.csv:wall_impulse_precav_rel_dev_vs_cole` |
| 2c | same, against the rigid-wall limit 2 p_m / 2 p_m theta (physically correct reference frame) | peak deviation 0; impulse ratio 0.155-0.293 | peak <15% | PASS | `case_matrix.csv:wall_peak_over_cole` |
| 2d | discretised-fluid FEM leg vs Taylor closed form (solver accuracy) | max deviation 0.5060 | <15% | FAIL | `chain_metrics.json:cases[*].fe_fluid` |
| 3 | system energy-conservation residual (whole shock window, worst level) | 0.0055 | <5e-2 | PASS | `fig5_energy_ledger.png`, `case_matrix.csv:energy_residual_worst_level` |
| 4a | SRS band coverage | 6 bands x 2 damping ratios | >=5 | PASS | `benchmarks/undex_shock/srs_comparison.json` |
| 4b | GCI (peak deflection / peak acceleration) | worst 0.0182 | <10% | PASS | `case_matrix.csv:gci_deflection`, `gci_acceleration` |
| 4c | GCI (pre-cavitation impulse) | worst 0.0990 | <10% | PASS | `case_matrix.csv:gci_impulse_precav` |
| 4d | criterion 4 combined (4a and 4b and 4c) | - | - | PASS | rows above |

**Adjudication note (no whitewash).** The literal reading of criterion 2 (FEM wall quantities must close with the Cole free-field quantities within 15%) is physically unsatisfiable: the wall pressure contains the reflection, so its peak is `2 p_m`; and the impulse is further truncated by cavitation (water cannot sustain tension).  Criterion 2 is therefore judged **FAIL** on 2a/2b, while 2c (deviation from `2 p_m`: 0) and 2d (solver accuracy: 0.5060) give the physically meaningful closure.  If a single binary verdict is required, criterion 2 = **FAIL**.

## 5. Caveats and limitations

1. **Anchor unobtainable.** No authorised copy of Cole (1948) exists in this environment (filesystem sweep, PDF corpus and network all failed; every attempt is recorded verbatim in `cole_anchor.json:acquisition_attempts`, and the network request was rejected outright by the sandbox security policy).  K, A, K_theta, K_b are therefore implementation parameters / untranscribed candidates with `usable_for_physics_validation = false`; criterion 1 only checks transcription fidelity.  Any claim of quantitative agreement with Cole's printed values would be out of scope.
2. **Cavitation not modelled.** The linear acoustic coupling lets water carry tension.  This report quantifies the defect: wall pressure reaches -1.1242 to -0.6699 MPa (tension) and cavitation starts at 99.3522-119.4458 us (about 0.3-0.7 theta).  Beyond that instant the model is non-physical, so the impulse is integrated only up to cavitation, and the pre-cavitation impulse has the worst grid convergence (4c).  The correct next step is a cavitation switch on the impedance term (detach when p_wall <= 0, cavity pressure zero), which makes the system nonlinear and needs per-step re-factorisation.
3. **The fluid leg is the exact normal plane-wave solution.** `p_wall = 2 p_i - rho c v_n` is exact only for normal incidence on a locally reacting surface; it carries no diffraction, spherical spreading, structural curvature or stiffeners.  Stiffened panels, oblique incidence and contact explosions need another model.  A discretised fluid (1-D acoustic FEM) appears only in the verification leg (120 elements, interface-pressure deviation 0.5060).
4. **Plate idealisation.** Hard simple supports, uniform thickness, no stiffeners, no initial imperfections, linear elastic (yield not checked: wall peaks of 11.5922 MPa are far above the 235 MPa mild-steel yield, so plasticity and rupture would occur in reality).
5. **Record length.** The window is 16 theta (about 3.67 ms), far shorter than the panel's first period (about 28.4 ms), so the 1-10 Hz SRS band is a quasi-static plateau and low-order structural participation has not developed; this is a window limitation, not an error.
6. **GCI observed order is clipped.** With small three-level differences the observed order is noisy (`chain_metrics.json:cases[*].gci` flags `order_clipped`); the p = 2 GCI is reported alongside, and both must be <10%% to pass.
7. **The criterion-2 deviation is not numerical error.** Because v(0) = 0 the wall peak equals 2 p_m exactly (identical on all three meshes), so its 'grid convergence' is an analytic identity rather than a convergence result, as marked in section 3.4.

## 6. Error judgement (vs reference, item by item; is it <3%%?)

| quantity | reference | deviation | <3%%? | note |
|---|---|---|---|---|
| areal-mass same-problem leg (deflection/velocity/acceleration/wall pressure) | Taylor closed form | <1%% (`tests/test_undex_chain.py::test_areal_mass_verification`) | yes | verifies coupling assembly and time integration |
| discretised-fluid leg interface pressure / velocity | Taylor closed form | 0.5060 / 0.0421 | no (within the 15% platform accuracy line) | linear-element dispersion of a short pulse (theta = 229 us, c theta = 0.344 m) |
| Cole recomputation | anchor records | 0 | yes | transcription fidelity only, not physical validation |
| FEM wall peak pressure | Cole free field p_m | 1.0000 | no | reflection doubling, physics not error |
| panel centre deflection | Taylor rigid areal-mass plate | ratio 0.9879-1.0430 | partly | panel-flexibility effect, not error |
| energy-conservation residual | 0 | 0.0055 | yes | whole shock window |

**Explicit answer to 'is it <3%?'**: only the same-problem verification leg (areal-mass plate vs Taylor closed form) and the Cole recomputation reach <3% (in fact floating-point / quadrature-error level); the discretised-fluid leg (0.5060) and the wall quantities vs the Cole free field (1.0000) do not, for the reasons located in section 5 (linear-element dispersion; reflection and cavitation physics).

## 7. Provenance (artifacts + commit + sha256)

| artifact | sha256 / note |
|---|---|
| `benchmarks/undex_shock/cole_anchor.json` | `cf8159f22306a6b9b5d6cceb4d50a3b52c6c5102c9ce513c068cccaf6e3b34ce` (status NOT_OBTAINED) |
| `src/tensorfem/undex_chain.py` | `14375eaa2cb70ac5caa25128c0def232533b4ba8a28acd008ec256e40a21fd73` |
| `src/tensorfem/undex_shock.py` (base module, taken from this repository at commit a706ec07) | `b1a5e4796ae4a01757005bea7ca8143b410bbf754a7a02b00c90be16281a8ec6` |
| `scripts/w23_undex_shock_chain_run.py` | `4f2133984ff9106868f5fa6ca482d3917538a29e3d90417c5e698f6b2f4e0dae` |
| `tests/test_undex_chain.py` | `c68838d98e3ec28c5fe2256154e871eb7107bfcaca4d1a6b1ad1ddce728c44c3` |
| `benchmarks/undex_shock/case_matrix.csv` | per-case metrics; source of every number above |
| `benchmarks/undex_shock/srs_comparison.json` | SRS curves and band-wise deviations |
| `benchmarks/undex_shock/chain_metrics.json` | GCI raw three-level values, bubble-scaling machine check, parameter provenance |
| `docs/assets/w23_undex_shock/*.png` | fig1_cole_environment.png, fig2_wall_pressure.png, fig3_panel_response.png, fig4_srs.png, fig5_energy_ledger.png, fig6_grid_convergence.png |

Figures: `fig1_cole_environment.png` (Cole pulses and bubble periods), `fig2_wall_pressure.png` (wall pressure vs Taylor / rigid-wall limit / Cole free field), `fig3_panel_response.png` (centre deflection and acceleration), `fig4_srs.png` (SRS with band edges), `fig5_energy_ledger.png` (incident / reflected / plate energy and residual), `fig6_grid_convergence.png` (GCI).
