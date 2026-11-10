# Load-spectrum residual life with crack-growth retardation (Wheeler) — verification report

Campaign node **IN55W3-N05** · repository `TensorFEM` · module
`tensorfem.fatigue.spectrum_retardation` · evidence
`artifacts/fatigue/paris_spectrum_retardation_evidence.json` · benchmark
`benchmarks/fatigue/paris_spectrum_retardation.py`.

This face is new relative to the two existing fatigue crack-growth faces:

| face | owns | explicitly does **not** do |
|---|---|---|
| `fatigue_crack_propagation` (legacy) | SIF library, constant-`Y` Paris integrator, single-pass spectrum walk | geometry factor discarded from the life integral |
| `fatigue.crack_growth_geometry` (P135W2-N03) | geometry-aware `Delta K(a)`, RK4 in `ln a`, `K_IC` root solve, GCI | *"no crack-closure/retardation (no overload, no Wheeler model)"* (verbatim scope statement) |
| **`fatigue.spectrum_retardation` (this node)** | ordered block load spectra, Irwin plastic zone, Wheeler shielding, spectrum residual life from an inspection threshold, inverse identification of `(C, m)` | no Elber `U`, no Willenborg residual stress, no overload-interaction decay, no `a/c` evolution |

Two engineering questions are added here: (i) how long does a detail last after a
crack is found, when the load history contains repeated tensile overloads, and
(ii) how large is that load-history correction compared with the geometry-factor
correction quantified by P135W2-N03.

## 1. Case introduction (physical background, geometry, parameters)

A transverse butt-weld toe on 25 mm hull plating carries a semi-elliptical
surface crack at the toe. NDT detects `a_0 = 2 mm` deep, `c_0 = 6 mm` long
(`a/c = 1/3`, frozen). The load history is block-wise: baseline cycles at
`Delta sigma = 120 MPa` (`R = 0.1`) plus a periodic tensile overload (one cycle
at `OLR x Delta sigma`) — the standard idealisation of a detail that sees
repeated heavy-weather or slamming events inside a long low-amplitude record.

| quantity | symbol | value | source |
|---|---|---|---|
| yield stress | `sigma_y` | 355 MPa | S355 hull structural steel |
| fracture toughness | `K_IC` | 60 MPa sqrt(m) | conservative weld / HAZ value |
| Paris coefficient | `C` | 5.5e-30 m/cycle/(Pa sqrt(m))^3 | ferritic steel, SI units |
| Paris exponent | `m` | 3 | ferritic steel |
| stress range | `Delta sigma` | 120 MPa | hot-spot range at the detail |
| stress ratio | `R` | 0.1 | `sigma_max = Delta sigma/(1-R)` is derived |
| IIW detail category | FAT | 90 MPa | transverse butt weld toe |
| Rayleigh RMS range | `sigma_rms` | 60 MPa | narrow-band spectrum case |

Three canonical ship details are assessed (`weld_toe_surface_crack`,
`hole_edge_crack`, `fillet_weld_toe_crack`); their geometry data (thickness, hole
radius, plate width, initial flaw sizes, validity bounds) live in `DETAILS` in
`src/tensorfem/fatigue/spectrum_retardation_qualification.py`.

## 2. Method

1. **Stress-intensity factors** — reused from the verified library
   (`k_semi_elliptical_surface_crack` Newman-Raju deepest point,
   `k_hole_edge_crack` Kirsch/Tada, `k_edge_crack_finite_width`), wrapped as
   `Delta K(a)` callables. The same callable is evaluated at `sigma_max` for
   `K_I`, so the critical-crack solve and the growth integral cannot drift apart.
2. **Plastic zone** — Irwin, `r_p = kappa (K_max/sigma_y)^2`,
   `kappa = 1/(6 pi)` plane strain, `1/(2 pi)` plane stress.
3. **Wheeler shielding** — `phi = (r_y,i/(a_p - a_i))^m`,
   `a_p = a_ol + r_y,overload`, clipped to `(0, 1]`; `phi = 1` exactly once the
   tip leaves the overload plastic zone. A zone is created whenever an
   overload-flagged cycle raises `K_max`.
4. **Growth integration** — block-wise classical RK4 in `t = ln a`
   (`d ln a/dN = phi C Delta K(a)^m / a`), `substeps_per_block` steps per block,
   plus a bracketed bisection (80 halvings) for the exact failure cycle inside
   the step that crosses `a_c`. Spectra are repeated until `a >= a_c`.
5. **Critical crack** — root of `K_I(a_c) = K_IC` inside the validity interval of
   the geometry factor; if the fit never reaches `K_IC`, the crack is reported at
   the section boundary (`critical_mode = "penetration"`) instead of
   extrapolating the fit.
6. **Cross-adjudication** — IIW FAT90 bilinear S-N at the same stress range, and
   a Palmgren-Miner life under the same Rayleigh band spectrum.
7. **Inverse identification** — `fit_paris_parameters` recovers `(C, m)` by least
   squares on `(ln Delta K, ln da/dN)` from central differences of an `a(N)`
   curve, so a prediction can be re-anchored on inspection data.
8. **Toolchain** — stdlib only in the physics module; `multiprocessing` pool for
   the independent qualification cases; `matplotlib` (Agg) for figures; `pytest`
   for the acceptance lane (`pytest --fast` = 32 tests in 15 s; the full
   qualification lane ≈ 40 s).

## 3. Results

### 3.1 Three details under one spectrum (1000 baseline cycles + one overload cycle, OLR 1.5)

| detail | unretarded `N` (cycles) | Wheeler `N` (cycles) | life factor | notes |
|---|---|---|---|---|
| `weld_toe_surface_crack` | 4.7826e+05 | 5.1975e+06 | 10.87 | unretarded 478260; legacy Y=1 (repeated spectra) 620301 (ratio 1.297); legacy helper one-pass output 1001 (reached_critical=False); a_c=0.0161659 (fracture_K_IC); min phi=0.08783; retarded cycles=5.191e+06 |
| `hole_edge_crack` | 1.1813e+05 | 1.2653e+06 | 10.71 | unretarded 118126; legacy Y=1 (repeated spectra) 1.46335e+06 (ratio 12.39); legacy helper one-pass output 1001 (reached_critical=False); a_c=0.0141185 (fracture_K_IC); min phi=0.08788; retarded cycles=1.263e+06 |
| `fillet_weld_toe_crack` | 2.4521e+05 | 2.6634e+06 | 10.86 | unretarded 245213; legacy Y=1 (repeated spectra) 969505 (ratio 3.954); legacy helper one-pass output 1001 (reached_critical=False); a_c=0.00460281 (fracture_K_IC); min phi=0.08783; retarded cycles=2.66e+06 |

### 3.2 Overload-ratio sweep (overload every 5000 cycles; each row paired with its own unretarded run)

| OLR | `N` Wheeler | `N` unretarded | life factor | `a_c` (mm) |
|---|---|---|---|---|
| 1.00 | 4.9066e+05 | 4.9066e+05 | 1 | 20.000 |
| 1.25 | 1.359e+06 | 4.9017e+05 | 2.773 | 19.824 |
| 1.50 | 5.0807e+06 | 4.7917e+05 | 10.6 | 16.166 |
| 2.00 | 2.5765e+07 | 4.5046e+05 | 57.2 | 11.521 |

### 3.3 Overload-spacing sweep (OLR 1.5)

| spacing (cycles) | `N` Wheeler | life factor vs unretarded |
|---|---|---|
| 1000 | 5.1975e+06 | 10.87 |
| 5000 | 5.0807e+06 | 10.6 |
| 20000 | 3.8972e+06 | 8.13 |
| 100000 | 6.6575e+05 | 1.389 |

### 3.4 Inverse identification of the Paris constants

Identified `C = 5.480128e-30` (generating `5.5e-30`),
`m = 3.000215` (generating `3.0`),
`R^2 = 0.9999999991` on 199
growth-rate points.

### 3.5 Acceptance criteria (all 25 machine-generated records)

| id | criterion | measured | oracle | metric | tolerance | verdict |
|---|---|---|---|---|---|---|
| `a1_infinite_plate` | K_I infinite plate vs closed form | 17724538.50905516 | 17724538.50905516 | 0 | 0.02 | PASS —  |
| `a2_centre_crack_finite_width` | secant correction vs exact collinear-array factor sqrt(W tan(pi a/W)) at 2a/W=0.2 | 18174889.268422306 | 18025529.013954245 | 0.008286 | 0.02 | PASS —  |
| `a3_edge_crack_limit` | edge-crack finite-width factor vs 1.12 sigma sqrt(pi a) as a/W -> 0 | 1984757.537118913 | 1985148.3130141783 | 0.0001968 | 0.02 | PASS —  |
| `a4_surface_crack_semicircular` | Newman-Raju semi-circular surface crack vs 0.6637 sigma sqrt(pi a) | 3713826.417593333 | 3720032.670323305 | 0.001668 | 0.02 | PASS — 0.6637 is the standard semi-circular edge-crack surface value |
| `b1_paris_integration_closed_form` | spectrum integrator (model='none', Y=1) vs closed-form Paris integral | 467133.6084605809 | 467133.6086495421 | 4.045e-10 | 0.01 | PASS — critical crack from K_I(a_c)=K_IC returned 0.010000 m |
| `c1_zero_delta_k` | da/dN at Delta K = 0 | 0.0 | 0.0 | 1 | 1e-12 | PASS —  |
| `c2_subthreshold_no_growth` | spectrum whose Delta K stays below the threshold grows nothing | False | False |  |  | PASS — life=inf (inf), events=1 |
| `c3_zero_amplitude_block_rejected` | a zero-amplitude block is rejected instead of silently growing a crack | True | True |  |  | PASS —  |
| `d1_initial_size_monotonicity` | N_residual strictly decreases with the detected initial crack size a0 | 5197465.303101374 | 1868033.046793382 | 0.6406 |  | PASS — a0 (m) -> N: 0.002:5.197e+06, 0.0025:4.214e+06, 0.003:3.497e+06, 0.004:2.514e+06, 0.005:1.868e+06 |
| `d2_initial_size_table` | residual life vs a0 table |  |  |  |  | PASS — a0=0.002 -> N=5.19747e+06; a0=0.0025 -> N=4.2139e+06; a0=0.003 -> N=3.49712e+06; a0=0.004 -> N=2.51363e+06; a0=0.005 -> N=1.86803e+06 |
| `e1_paris_vs_sn_constant_amplitude` | Paris constant-amplitude life vs IIW FAT90 S-N life at the same stress range | 490655.1344693342 | 843750.0 | -0.2354 | 1 | PASS — metric is log10(N_paris / N_sn) |
| `e2_paris_vs_miner_rayleigh` | Paris spectrum life vs Palmgren-Miner S-N life under the same Rayleigh band spectrum | 119993.44425375231 | 224509.40437962653 | -0.2721 | 1 | PASS — metric is log10(N_paris / N_miner) |
| `f1_growth_integral_gci` | Richardson GCI of the spectrum growth integral (substeps/block 4/8/16) | 5080716.40755089 | 5080716.268889634 | 3.411e-08 | 0.05 | PASS — observed order 0.50; lives (4/8/16) = 5.08072e+06/5.08072e+06/5.08072e+06 |
| `f2_rayleigh_band_gci` | Richardson GCI of the Rayleigh band discretisation (4/8/16 bands) | 119995.10873573639 | 119997.46292215811 | 2.452e-05 | 0.05 | PASS — observed order 0.77; lives = 119991/119993/119995 |
| `g1_exponent_sweep` | life strictly decreases with the Paris exponent m at iso-rate reference Delta K | 780450.8232376014 | 350132.5925033275 |  |  | PASS — m -> N: 2:7.805e+05, 3:4.907e+05, 4:3.501e+05 |
| `h1_wheeler_bounds` | phi in (0,1] on the whole shielding interval, non-decreasing, exactly 1 past a_p | 0.08779149519890295 | 1.0 |  |  | PASS — a_p = 0.0041762 m, r_p = 0.000176195 m, phi_min = 0.0877915, terminal phi = 1 |
| `h2_wheeler_overload_anchor` | phi at the overload crack size equals OLR^(-2m) (radius ratio K_base/K_OL squared, cubed) | 0.08779149519890295 | 0.0877914951989026 | 3.952e-15 | 1e-09 | PASS — overload spacing 1000 cycles, OLR = 1.5 |
| `i1_overload_ratio_sweep` | retarded life increases monotonically with the overload ratio | 1.0 | 10.603228525601475 |  |  | PASS — OLR -> life factor (paired with its own unretarded life): 1:1 (4.907e+05/4.907e+05), 1.25:2.7725 (1.359e+06/4.902e+05), 1.5:10.603 (5.081e+06/4.792e+05) |
| `i2_unity_overload_coincidence` | OLR = 1 reproduces the unretarded life exactly and creates no zone | 1.0 | 1.0 | 0 | 1e-09 | PASS — retarded cycles at OLR=1: 0, zones created: 64 (unity overload re-creates a zone at the current tip, but phi = 1 there by construction) |
| `j1_no_overload_limit_vs_independent_quadrature` | wheeler-enabled solver on an overload-free spectrum vs independent ln-a trapezoid of dN/da | 490655.13452035654 | 490657.85933655506 | 5.553e-06 | 0.01 | PASS — model='none' control: 490655 cycles |
| `k1_spacing_sweep` | life factor decreases as overloads are spaced beyond the plastic-zone crossing time | 10.592909230885086 | 1.3568504759437692 |  |  | PASS — spacing -> factor: 1000:10.593, 20000:7.9428, 100000:1.3569 |
| `l1_paris_parameter_recovery` | recovered (C, m) from a synthetic a(N) curve vs the generating constants | 3.0000540935025404 | 3.0 | 1.803e-05 | 0.01 | PASS — C_fit=5.495003e-30 (rel err 9.09e-04), m_fit=3.000054, R^2=1.00000000, n=399 |
| `m1_weld_toe_surface_crack` | residual life for weld_toe_surface_crack under one spectrum (1000 baseline cycles + OLR 1.5) | 5197465.303101374 | 478259.6804066317 | 10.87 |  | PASS — unretarded 478260; legacy Y=1 (repeated spectra) 620301 (ratio 1.297); legacy helper one-pass output 1001 (reached_critical=False); a_c=0.0161659 (fracture_K_IC); min phi=0.08783; retarded cycles=5.191e+06 |
| `m2_hole_edge_crack` | residual life for hole_edge_crack under one spectrum (1000 baseline cycles + OLR 1.5) | 1265263.779354929 | 118126.25481985944 | 10.71 |  | PASS — unretarded 118126; legacy Y=1 (repeated spectra) 1.46335e+06 (ratio 12.39); legacy helper one-pass output 1001 (reached_critical=False); a_c=0.0141185 (fracture_K_IC); min phi=0.08788; retarded cycles=1.263e+06 |
| `m3_fillet_weld_toe_crack` | residual life for fillet_weld_toe_crack under one spectrum (1000 baseline cycles + OLR 1.5) | 2663354.180702483 | 245213.09238257294 | 10.86 |  | PASS — unretarded 245213; legacy Y=1 (repeated spectra) 969505 (ratio 3.954); legacy helper one-pass output 1001 (reached_critical=False); a_c=0.00460281 (fracture_K_IC); min phi=0.08783; retarded cycles=2.66e+06 |

## 4. Figures

| figure | content |
|---|---|
| `docs/assets/fatigue_spectrum_retardation/wheeler_shielding_profile.png` | `phi(a)` after one overload for OLR 1.25 / 1.5 / 2.0; shielding is exhausted exactly at `a_p` |
| `docs/assets/fatigue_spectrum_retardation/life_factor_sweeps.png` | life factor vs overload ratio and vs overload spacing |
| `docs/assets/fatigue_spectrum_retardation/residual_life_vs_detection_threshold.png` | `N_residual(a_0)` for weld toe and open hole, Wheeler vs unretarded |
| `docs/assets/fatigue_spectrum_retardation/paris_parameter_identification.png` | `da/dN` vs `Delta K` with the generating and identified Paris lines |
| `docs/assets/fatigue_spectrum_retardation/criteria_utilisation.png` | every tolerance-bearing criterion as `|metric|/tolerance` |

## 5. Pitfalls, limitations and applicability

- **Wheeler has no closure-memory decay.** A periodic overload sequence whose
  spacing is shorter than the plastic-zone crossing life re-creates the zone at
  the current tip every repetition, so shielding never relaxes: spacing 1000 →
  factor 10.87, spacing 100000 →
  1.39. The model therefore
  *upper-bounds* retardation. The OLR 2.0 value
  (57.2x, OLR 2.0) is far outside the
  1.2-1.5x range usually reported for a single overload and must not be booked
  as design credit without a closure-based model (NASGRO / Willenborg) or test
  data.
- **The load-history correction is second order next to the geometry-factor
  correction.** For the same spectrum, the legacy `Y = 1` integration is
  non-conservative by a factor 1.297 (weld toe),
  12.39 (hole edge) and
  3.954 (fillet toe), while the Wheeler correction
  is a model-dependent factor of order 1-10. Geometry first, closure second.
- **The legacy spectrum helper cannot express a residual life at all.**
  `paris_integrate_variable_amplitude` stops after one spectrum pass and returns
  1001 cycles with `reached_critical = False` for every detail above; a residual
  life needs spectrum repetition, which this module adds.
- **Distribution spectra need a pinned physical peak.** The band set is
  moment-matched (`sum p_k Delta sigma_k^3 = E[Delta sigma^3]` to round-off), so
  refining the bands does not change the growth integral; without pinning
  `critical_stress_pa`, band refinement also moves `a_c` through the largest
  band range and the two effects cannot be separated (that confusion produced a
  26% spurious GCI before the peak was pinned).
- **Scope**: LEFM, small-scale yielding, mode I, `Delta K > Delta K_th`,
  constant amplitude within a block, frozen `a/c`, no residual stress, no
  compressive-overload effects, no overload interaction, no crack-tip
  blunting/re-sharpening. `phi` is sampled at substep boundaries, so
  `min_retardation_factor` is a sampled minimum.
- **Numerics**: growth rates span decades, hence RK4 in `ln a`; a uniform step in
  `a` does not converge for these geometries. RK4 nodes are clamped to the
  geometry-factor validity bound so an overshoot cannot raise a spurious
  extrapolation error.

## 6. Error judgement

| quantity | criterion id | deviation | acceptance | reconciled |
|---|---|---|---|---|
| (a) SIF 锚：有限宽中心裂纹 vs 精确共线阵因子 | `a2_centre_crack_finite_width` | 0.83 % | ≤ 2 % | 是 |
| (b) Paris 积分：谱积分器 vs 闭式解 | `b1_paris_integration_closed_form` | 4.0e-8 % | ≤ 1 % | 是 |
| (e) Paris 恒幅寿命 vs IIW S-N | `e1_paris_vs_sn_constant_amplitude` | 0.235 个数量级 | ≤ 1 个数量级 | 是 |
| (e) Paris 谱寿命 vs Palmgren-Miner | `e2_paris_vs_miner_rayleigh` | 0.272 个数量级 | ≤ 1 个数量级 | 是 |
| (f) 生长积分分步 GCI | `f1_growth_integral_gci` | 3.4e-6 % | ≤ 5 % | 是 |
| (f) Rayleigh 分带 GCI | `f2_rayleigh_band_gci` | 2.5e-3 % | ≤ 5 % | 是 |
| (h) Wheeler 锚 φ(a_ol) = OLR^(−2m) | `h2_wheeler_overload_anchor` | 4.0e-13 % | ≤ 1e-7 % | 是 |
| (j) 无过载极限 vs 独立 ∫dN/da 梯形积分 | `j1_no_overload_limit_vs_independent_quadrature` | 5.6e-4 % | ≤ 1 % | 是 |
| (l) Paris 常数反演 (C, m) | `l1_paris_parameter_recovery` | 1.8e-3 % | ≤ 1 % | 是 |

**Are the key deviations below 3 %?** Yes, for every quantity that has an
oracle: SIF anchors within 0.83 %, constant-amplitude Paris integral 4e-8 %
(bisection resolution), no-overload limit vs an independent `ln a` trapezoid
5.6e-4 %, Wheeler anchor vs `OLR^(-2m)` 4e-13 %, identified Paris constants
1.8e-3 %, and both GCI values (3.4e-6 % and 2.5e-3 %) far inside the 5 %
requirement. The only criteria that are not error-free are the by-design
order-of-magnitude S-N cross-adjudications (0.235 and 0.272 decades against a
one-decade tolerance) and the monotonicity/sweep criteria, which have no
reference value.

The physical residual-life numbers are **model predictions, not measurements**:
no experimental `N_f` exists for these idealised details in this repository, so
the 3 % statement covers the numerical/analytic verification chain only, not
agreement with test data. The retardation magnitude is conditional on the
Wheeler assumption, as stated in section 5 — reported as an honest limitation,
not covered up.
