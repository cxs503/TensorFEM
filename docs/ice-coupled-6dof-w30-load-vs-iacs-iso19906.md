# ice-coupled-6dof (W30): random-ice 6-DOF load ensemble vs IACS UR I2 / ISO 19906

- task: `ice-coupled-6dof` (mission 21515), repository TensorFEM, commit `c400ea246197966d0009b27fab1e8f4a2bbcbe32`
- artefact: `papers/artifacts/w30/ice-coupled-6dof/ice_load_vs_iso19906.json` (generated 2026-11-15T07:08:03Z)
- anchor artefact: `benchmarks/anchors/iacs-ur-i-ice-load.json` (sha256 `3b4ec36679e3a882cd3b1f9027fed68526791a001ae5850edc66da7a82ad3271`), 17 clause anchors

## 1. Case and parameters

Rigid 6-DOF hull (surge, sway, heave, roll, pitch, yaw) with constant added mass, hydrostatic restoring in heave/roll/pitch and Cummins radiation memory kernels: L = 100.0 m, B = 20.0 m, T = 6.0 m, Cb = 0.7, displacement = 8.61 kt. These are the declared `ShipModel` particulars of the W23 chain, not a validated ship.

| group | ice concentration | mean thickness [m] | median floe diameter [m] | seeds |
|---|---|---|---|---|
| G1-mild-c0.3 | 0.3 | 0.6 | 3.0 | 24 |
| G2-moderate-c0.5 | 0.5 | 0.6 | 3.0 | 24 |
| G3-severe-c0.7 | 0.7 | 0.6 | 3.0 | 24 |
| G4-design-severity | 0.7 | 1.2 | 10.0 | 24 |

dt = 0.01 s, duration = 40.0 s, commanded speed = 5.0 m/s, seeds = 7101..7124 (24 per group = 96 runs).

## 2. Method and tool chain

1. **Anchor transcription** (`benchmarks/anchors/iacs_ur_i_ice_load_anchor.py`): the unaltered IACS PDFs in `benchmarks/anchors/sources/iacs/` are text-extracted; each anchor is the exact substring between two textual markers on one page, hashed (`sha256_file`, `sha256_quote`), and every constant used by the design-load routine is re-parsed out of those quotes and machine-compared with the Python literal.
2. **Ensemble** (`benchmarks/ice_coupled_6dof/run_w30_ice_load_vs_code.py`): random floe field (Poisson arrivals with rate n(B+D)V, lognormal diameters, Gaussian thicknesses, random azimuth and drift) advanced against the hull with velocity-Verlet; for every impact cluster the peak contact force, peak contact area and peak contact pressure are recorded. The contact law is the ISO 19906 pressure--area form with declared coefficients.
3. **Comparison**: IACS UR I2 bow design load from the transcribed clauses (per Polar Class), plus distribution fits (Weibull MLE, Gumbel) with KS/AIC and tail extrapolation.

### Anchored clauses (each with sha256)

| anchor | clause | what it carries | page | sha256(quote) |
|---|---|---|---|---|
| `IACS-UR-I2-I2.3.1(ii)` | I2.3.1 (ii) | definition of the design ice load: average pressure over a rectangular patch b x w | 3 | `030ca59640823fee...` |
| `IACS-UR-I2-I2.3.1(iv)` | I2.3.1 (iv) | fixed load patch aspect ratio for non-bow areas (AR = 3.6) | 3 | `2e98f66099d3d778...` |
| `IACS-UR-I2-I2.3.2.1(iii)(a)` | I2.3.2.1 (iii) (a) | bow shape coefficient fai = min(fai1, fai2, fai3) and its constants | 5 | `406bf63a9b021873...` |
| `IACS-UR-I2-I2.3.2.1(iii)(b)` | I2.3.2.1 (iii) (b) | bow glancing impact force Fi = fai CFC DUI^0.64 [MN] (displacement exponent 0.64) | 5 | `1ee9e4164af59050...` |
| `IACS-UR-I2-I2.3.2.1(iii)(c)` | I2.3.2.1 (iii) (c) | load patch aspect ratio ARi = 7.46 sin(beta'i) >= 1.3 | 5 | `7456c405821674b3...` |
| `IACS-UR-I2-I2.3.2.1(iii)(d)` | I2.3.2.1 (iii) (d) | line load Qi = Fi^0.61 CFD / ARi^0.35 [MN/m] | 5 | `4477c351c4128e86...` |
| `IACS-UR-I2-I2.3.2.1(iii)(e)` | I2.3.2.1 (iii) (e) | design pressure Pi = Fi^0.22 CFD^2 ARi^0.3 [MPa] (pressure-area trend) | 5 | `a2d824e02ce5a1ed...` |
| `IACS-UR-I2-I2.3.2.2(i)(a)` | I2.3.2.2 (i) (a) | non-bow force FNonBow = 0.36 CFC DF [MN] | 6 | `b647055c7acbcc68...` |
| `IACS-UR-I2-I2.3.2.2(i)(b)` | I2.3.2.2 (i) (b) | non-bow line load QNonBow = 0.639 FNonBow^0.61 CFD [MN/m] | 6 | `aa78aa810e50b740...` |
| `IACS-UR-I2-I2.3.2.2(i)-DF` | I2.3.2.2 (i), definition of DF | ship displacement factor DF = DUI^0.64 below CFDIS, linear above | 6 | `8dd90b40b462136a...` |
| `IACS-UR-I2-I2.3.3(i)` | I2.3.3 (i) | bow design load patch: wBow = FBow/QBow, bBow = QBow/PBow | 6 | `aa5dd6cf79a68f82...` |
| `IACS-UR-I2-I2.3.3(ii)` | I2.3.3 (ii) | non-bow design load patch: wNonBow = FNonBow/QNonBow, bNonBow = wNonBow/3.6 | 7 | `2fdb029b0fa04bff...` |
| `IACS-UR-I2-I2.3.4(i)` | I2.3.4 (i) | average pressure over the design patch Pavg = F/(b w) [MPa] | 7 | `5fa7a293f7a6eb26...` |
| `IACS-UR-I2-Table3` | I2.3.4 (ii), Table 3 | peak pressure factors PPFp, PPFt, PPFs (local pressure concentration) | 7 | `68dcd0bd9a503bc4...` |
| `IACS-UR-I2-Table1` | I2.3.2.1 (iii), Table 1 | class factors CFC, CFF, CFD, CFDIS, CFL for PC1..PC7 | 4 | `54724517e922026e...` |
| `IACS-UR-I2-Table2` | I2.3.2.1 (iv), Table 2 | class factors CFCV, CFQV, CFPV for vertical-sided bows (PC6, PC7) | 4 | `25637e13c617b87b...` |
| `IACS-UR-I2-Table4` | I2.3.5, Table 4 | hull area factors AF per hull area and Polar Class | 8 | `83f52a98d187fd7d...` |

Source documents: `UR-I2Rev5.pdf` (IACS Req. 2006/Rev.5 2025, sha256 `d107c3b00dfd411f215c654252b978ae957b9b2d16bcefb10d3c42e410518513`, ? pp, https://iacs.s3.af-south-1.amazonaws.com/wp-content/uploads/2025/06/11113711/UR-I2Rev5.pdf); `UR-I1Rev2-Corr1.pdf` (IACS Req. 2007/Rev.2 2022 (Corr.1), sha256 `26de20c4216d2e546f86d03cd614a5dfef7394a220dc3fc9ed0e5cb85cee5a8f`, ? pp, https://iacs.s3.af-south-1.amazonaws.com/wp-content/uploads/2022/02/12082733/ur-i1rev2-1.pdf).

ISO 19906 status: **WITHHELD** — no licensed copy of ISO 19906 is reachable from this work tree and the standard is not public; writing a recalled coefficient would be fabrication. Declared coefficients used downstream: C_R = 1.0 MPa, A_0 = 1.0 m2, ex = -0.28 (grade `DECLARED_REFERENCE_NOT_NORMATIVE`).

## 3. Results

### 3.1 Ensemble peak-load statistics (per impact)

| group | impacts | mean F [MN] | sd F [MN] | p95 F [MN] | max F [MN] | mean p [MPa] | max p [MPa] | code area / mean area |
|---|---|---|---|---|---|---|---|---|
| G1-mild-c0.3 | 3640 | 0.2435 | 0.1126 | 0.4420 | 0.9285 | 1.885 | 8.501 | 10.1 |
| G2-moderate-c0.5 | 6069 | 0.2402 | 0.1078 | 0.4295 | 0.7835 | 1.886 | 7.689 | 10.4 |
| G3-severe-c0.7 | 8497 | 0.2377 | 0.1079 | 0.4285 | 0.9896 | 1.898 | 8.921 | 10.5 |
| G4-design-severity | 985 | 1.0822 | 0.4922 | 1.9404 | 3.9012 | 1.055 | 4.620 | 1.3 |

### 3.2 Convergence on the seed axis (criterion 2)

| group | full-mean F [MN] | last-10-seed mean F [MN] | relative difference | <=5%? |
|---|---|---|---|---|
| G1-mild-c0.3 | 0.2434 | 0.2436 | +0.08% | yes |
| G2-moderate-c0.5 | 0.2402 | 0.2384 | -0.77% | yes |
| G3-severe-c0.7 | 0.2377 | 0.2364 | -0.51% | yes |
| G4-design-severity | 1.0847 | 1.0579 | -2.47% | yes |

### 3.3 Distribution fits and tail assumption (criterion 4)

| group | Weibull k | Weibull KS p | Gumbel scale | Gumbel KS p | preferred (AIC) | extrapolated q(1/N) [MN] | observed max [MN] |
|---|---|---|---|---|---|---|---|
| G1-mild-c0.3 | 2.25 | 0.000 | 0.185 | 0.000 | weibull | 0.698 | 0.929 |
| G2-moderate-c0.5 | 2.33 | 0.000 | 0.182 | 0.000 | weibull | 0.685 | 0.783 |
| G3-severe-c0.7 | 2.30 | 0.000 | 0.180 | 0.000 | weibull | 0.697 | 0.990 |
| G4-design-severity | 2.28 | 0.021 | 0.821 | 0.000 | weibull | 2.840 | 3.901 |

Tail assumption: per-impact peaks treated as i.i.d.; the ensemble is clustered inside each floe field, so the fitted tail is optimistic. The extrapolated 1/N quantile is compared with the observed maximum in the table above — that column pair *is* the check.

### 3.4 Comparison with the code (criterion 3 and 5)

| group | item | measured | reference | deviation | <15%? | cause class |
|---|---|---|---|---|---|---|
| G1-mild-c0.3 | C1-force-mean | 0.244 MN | 4.697 | -94.8% | no | formulation_scope |
| G1-mild-c0.3 | C2-force-seedmax | 0.613 MN | 4.697 | -87.0% | no | formulation_scope |
| G1-mild-c0.3 | C3-pressure-peak-vs-Pavg | 8.501 MPa | 3.116 | +172.8% | no | model_difference |
| G1-mild-c0.3 | C3b-pressure-matched-band | n/a MPa | 3.116 | n/a | not evaluated | not_evaluated |
| G1-mild-c0.3 | C4-iso-closure | 1.885 MPa | 1.705 | +10.6% | yes | closure_tautology |
| G1-mild-c0.3 | C5-exponent-vs-iso | -0.280 - | -0.280 | -0.0% | yes | closure_tautology |
| G1-mild-c0.3 | C6-exponent-vs-code | -0.280 - | 0.282 | -199.3% | no | formulation_scope |
| G2-moderate-c0.5 | C1-force-mean | 0.240 MN | 4.697 | -94.9% | no | formulation_scope |
| G2-moderate-c0.5 | C2-force-seedmax | 0.645 MN | 4.697 | -86.3% | no | formulation_scope |
| G2-moderate-c0.5 | C3-pressure-peak-vs-Pavg | 7.689 MPa | 3.116 | +146.7% | no | model_difference |
| G2-moderate-c0.5 | C3b-pressure-matched-band | n/a MPa | 3.116 | n/a | not evaluated | not_evaluated |
| G2-moderate-c0.5 | C4-iso-closure | 1.886 MPa | 1.716 | +9.9% | yes | closure_tautology |
| G2-moderate-c0.5 | C5-exponent-vs-iso | -0.280 - | -0.280 | +0.0% | yes | closure_tautology |
| G2-moderate-c0.5 | C6-exponent-vs-code | -0.280 - | 0.282 | -199.3% | no | formulation_scope |
| G3-severe-c0.7 | C1-force-mean | 0.238 MN | 4.697 | -94.9% | no | formulation_scope |
| G3-severe-c0.7 | C2-force-seedmax | 0.650 MN | 4.697 | -86.2% | no | formulation_scope |
| G3-severe-c0.7 | C3-pressure-peak-vs-Pavg | 8.921 MPa | 3.116 | +186.3% | no | model_difference |
| G3-severe-c0.7 | C3b-pressure-matched-band | n/a MPa | 3.116 | n/a | not evaluated | not_evaluated |
| G3-severe-c0.7 | C4-iso-closure | 1.898 MPa | 1.722 | +10.2% | yes | closure_tautology |
| G3-severe-c0.7 | C5-exponent-vs-iso | -0.280 - | -0.280 | +0.0% | yes | closure_tautology |
| G3-severe-c0.7 | C6-exponent-vs-code | -0.280 - | 0.282 | -199.3% | no | formulation_scope |
| G4-design-severity | C1-force-mean | 1.082 MN | 4.697 | -77.0% | no | formulation_scope |
| G4-design-severity | C2-force-seedmax | 2.279 MN | 4.697 | -51.5% | no | formulation_scope |
| G4-design-severity | C3-pressure-peak-vs-Pavg | 4.620 MPa | 3.116 | +48.2% | no | model_difference |
| G4-design-severity | C3b-pressure-matched-band | 0.929 MPa | 3.116 | -70.2% | no | model_difference |
| G4-design-severity | C3c-force-matched-band | 1.259 MN | 4.697 | -73.2% | no | model_difference |
| G4-design-severity | C4-iso-closure | 1.055 MPa | 0.955 | +10.4% | yes | closure_tautology |
| G4-design-severity | C5-exponent-vs-iso | -0.280 - | -0.280 | +0.0% | yes | closure_tautology |
| G4-design-severity | C6-exponent-vs-code | -0.280 - | 0.282 | -199.3% | no | formulation_scope |

Code side (PC6, DUI = 8.61 kt): design force 4.697 MN, design pressure 3.116 MPa, patch 1.507 m2 (b = 0.549 m, w = 2.743 m). IACS-implied pressure--area exponent: 0.282.

## 4. Criteria

| criterion | statement | value | verdict |
|---|---|---|---|
| 1-anchors | >= 3 normative anchors, each with clause locator, verbatim quote and sha256 | 17 | PASS |
| 2-ensemble-convergence | last 10 seeds agree with the full ensemble mean within 5% | G1-mild-c0.3=0.0007593145151476545; G2-moderate-c0.5=-0.007664591988179328; G3-severe-c0.7=-0.005145608309004177; G4-design-severity=-0.02465060959093202 | PASS |
| 3-deviation-table | per-item deviation vs the code with an explicit '<15%?' answer and a cause classification for every item | items=18; within_15pct=0; not_evaluated=3; all_within_15pct=False | PASS |
| 4-distribution-fit | Weibull/Gumbel fit with goodness-of-fit and a stated tail assumption | G1-mild-c0.3=weibull; G2-moderate-c0.5=weibull; G3-severe-c0.7=weibull; G4-design-severity=weibull | PASS |
| 5-per-item-answer | answer 'is it < 15%' for every item and classify the cause | anchored_within_15pct=0; anchored_total=18; closure_within_15pct=8; closure_total=8 | PASS |

**15% deviation gate over anchored items: FAIL.** Every anchored item is outside 15%; the causes are classified item by item as `formulation_scope` (the IACS design load is a prescribed glancing impact of a large floe, the ensemble is a random small-floe field), `model_difference` (same quantity, different model/scale) or `closure_tautology` (the ISO law is inside the simulation, so agreement only proves the implementation).

## 5. Pitfalls, limitations, scope

- not claimed: compliance with ISO 19906
- not claimed: validation of the IACS design load against model tests
- not claimed: extrapolation of the 6-DOF chain to a real ship
- items tagged closure_tautology compare the simulation against the law that generates its contact force; they can only prove the implementation, never the standard
- no ISO 19906 number is transcribed; all ISO-derived values use declared coefficients and are non_normative
- no hull geometry file is used, so Cb/L/B are the declared ShipModel particulars and are not independently validated here
- IACS UR I2 is transcribed at Rev.5 (June 2025); another revision changes the class factors and would move every comparison below
- only the bow-area chain (I2.3.2.1 (iii)) is evaluated; the vertical-side PC6/PC7 chain (I2.3.2.1 (iv)) and the non-bow chain (I2.3.2.2) are transcribed but not evaluated here
- ISO 19906 stays WITHHELD: the ISO-side comparison is against declared coefficients and cannot claim compliance with the standard
- IACS design loads are scantling loads for a prescribed glancing bow impact, not a time-domain simulation output, so agreement/disagreement with a 6-DOF ensemble is a formulation-scope comparison, not a validation of either side
- the area scale separation is explicit: the code patch is 1.507 m2 while the G1-G3 peak contact areas are ~0.02-0.03 m2; only G4 overlaps the code patch area within a factor 2, and there the matched-band items are the only transport-free pressure comparisons.

## 6. Error judgement (vs baseline)

| quantity | baseline | deviation | <3%? |
|---|---|---|---|
| ISO closure arm: measured ln p vs ln A slope | declared ex = -0.28 | see C5 item (closure, expected ~0) | yes by construction (closure, not validation) |
| C1-force-mean (G1) | IACS PC6 design value 4.697 | -94.8% | no |
| C3-pressure-peak-vs-Pavg (G1) | IACS PC6 design value 3.116 | +172.8% | no |

There is **no experimental benchmark** for the coupled ice-load level in this chain, so the 3% question can only be answered against the code value (above, not met) and against the closure arm (met by construction, but that is an implementation check). An honest statement is: the 3% criterion is **not met** for the physics comparison and the 15% gate is **not met** either; the reason is a formulation-scope gap, quantified below.

## 7. Figures

![fig1_peak_force_distribution.png](figures/fig1_peak_force_distribution.png)

*Per-impact peak ice load distributions per parameter group with Weibull/Gumbel fits and the IACS PC6 design force.*

![fig2_pressure_area.png](figures/fig2_pressure_area.png)

*Pressure--area plane: ensemble peaks, the declared ISO 19906 law (closure arm) and the IACS PC6 design point with the code-implied trend.*

![fig3_convergence.png](figures/fig3_convergence.png)

*Seed-axis convergence of the running mean peak force against the +/-5% band.*

![fig4_timeseries.png](figures/fig4_timeseries.png)

*Time-domain hull-ice contact force for one representative seed per group.*

![fig5_deviations.png](figures/fig5_deviations.png)

*Per-item deviation from the IACS design values with the +/-15% gate.*

(figure files: `papers/artifacts/w30/ice-coupled-6dof/figures/`)
