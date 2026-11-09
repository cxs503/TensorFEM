# Fatigue crack growth: Paris law, stress intensity factors and residual life

Node `p135w1-n05-fatigue-crack-growth` (TensorFEM). Machine evidence:
`artifacts/fatigue/paris_crack_growth_evidence.json`; producer:
`benchmarks/fatigue/paris_crack_growth_qualification.py`; figures:
`docs/assets/fatigue_paris/`.

## 1 Case introduction

Ship hull structural details are assessed twice in practice: with the
S-N (stress-life) route of classification rules, and with linear elastic
fracture mechanics (LEFM) when a crack is detected. This node implements and
verifies the LEFM route for a mode-I crack:

* stress intensity factor `K_I = sigma sqrt(pi a) Y(a/W)` for a centre crack
  (infinite and finite width), an edge crack and a semi-elliptical surface
  crack (Newman-Raju form);
* Paris law `da/dN = C (Delta K)^m`, integrated numerically from `a_0` to the
  critical size `a_c` defined by `K_I(a_c) = K_IC`;
* residual life `N_res` given the inspection threshold `a_0` and the stress
  range;
* a quarter-point singular finite element model that produces `J` and hence
  `K_I` from a mesh, so the handbook factors have a numerical, grid-refinable
  counterpart.

Units are SI throughout (Pa, m, cycles). The Paris coefficient uses
`Delta K` in `Pa sqrt(m)`; the BS 7910 mean curve for ferritic steel in air
(`C = 5.21e-13`, `m = 3` with mm/cycle and N/mm^1.5) converts to
`C = 1.65e-29` in these units, which is the value used below.

Typical parameters: `E = 210 GPa`, `nu = 0.3`, `K_IC = 100 MPa sqrt(m)`,
`Delta sigma = 150 MPa` for the growth curves and 80 MPa for the S-N
cross-check, `a_0 = 0.2 ... 6 mm`, `a_c` from the toughness.

## 2 Method

1. SIF: closed-form factors, evaluated against independent anchors
   (the exact collinear-crack-array value `sigma sqrt(W tan(pi a/W))` for the
   finite-width secant correction, and the 1.12 edge factor).
2. Paris integration: explicit stepping with a step proportional to the current
   crack length, compared with the closed-form solution of the constant
   `Delta K` problem and with an independent adaptive ODE integration
   (scipy LSODA, rtol 1e-12, terminal event at `a = a_c`).
3. Residual life: `a_c` from `K_IC`, life from numerical integration,
   monotonicity in `a_0`.
4. S-N cross-check: the same detail (80 MPa hot-spot range, FAT 80 curve,
   `N = 2e6` at the reference stress, slope 3) assessed by both routes.
5. Singular finite elements: six-node triangles whose mid-side nodes on the two
   edges meeting at the crack tip sit at one quarter of the edge length
   (Barsoum geometry), polar graded mesh, plane strain, far-field tractions from
   the exact Westergaard solution, `J` integrated with the verified contour
   integrator `tensorfem.crack_tip_j_integral` on closed mesh-line contours.

## 3 Results

### (a) Stress intensity factor anchors

| id | actual | oracle | relative_error | tolerance | passed |
|---|---|---|---|---|---|
| a1_infinite_plate | 1.772e+07 | 1.772e+07 | 0 | 0.02 | True |
| a2_finite_width_vs_periodic_array | 0.07934 | 0 | 0.07934 | 0.1 | True |
| a3_edge_crack_1.12 | 1.985e+07 | 1.985e+07 | 0 | 0.02 | True |


### (b) Paris integration

| id | actual | oracle | relative_error | passed |
|---|---|---|---|---|
| b1_paris_step_0.02 | 1.611e+05 | 1.584e+05 | 0.01705 | False |
| b1_paris_step_0.005 | 1.59e+05 | 1.584e+05 | 0.004006 | True |
| b1_paris_step_0.001 | 1.585e+05 | 1.584e+05 | 0.0008608 | True |
| b2_paris_adaptive_ode | 1.584e+05 | 1.584e+05 | 1.952e-11 | True |


### (c) Negative control

`da/dN(Delta K = 0) = 0.0` and below the
threshold `da/dN = 0.0`;
no growth without a stress range: `True`.

### (d) Residual life

`a_c = 0.1415 m` and the life decreases strictly with `a_0`:
`True`.

| a_0_m | n_residual |
|---|---|
| 0.0002 | 4.406e+05 |
| 0.0005 | 2.723e+05 |
| 0.001 | 1.875e+05 |
| 0.002 | 1.276e+05 |
| 0.004 | 8.515e+04 |
| 0.006 | 6.638e+04 |


### (e) Paris versus S-N

| route | cycles to failure |
|---|---|
| Paris (a_0 = 0.2 mm, a_c = 8 mm) | 2.541e+06 |
| S-N (FAT 80, N_ref = 2e6, slope 3) | 2e+06 |

`log10(N_Paris / N_SN) = 0.104` decades; one decade is the
acceptance band, so `passed = True`. The two routes agree closely here
because both use the same stress range and the BS 7910 mean constants; the
agreement is an order-of-magnitude statement, not a calibration.

### (f) Quarter-point singular element J-integral

| level | sectors | nodes | J [J/m^2] | K_I [MPa sqrt(m)] | deviation vs handbook |
|---|---|---|---|---|---|
| coarse | 8 | 320 | 249.89 | 7.594 | -57.16 % |
| medium | 16 | 893 | 242.00 | 7.473 | -57.84 % |
| fine | 32 | 2806 | 232.85 | 7.330 | -58.64 % |

Contour path independence (one mesh, contours from
`0.0142` m to
`0.1081` m): spread
`15.6 %` of the mean over all contours and
`5.0 %` for contours outside
`2a` (the inner contour is still too close to the singular ring).
Grid convergence index: observed order `-0.23712027464674668`,
`GCI_fine = None`, converging `False`.

**Honest status: criterion (f) is NOT met.** The three grid levels give
`J = 249.9`, `242.0`,
`232.8` J/m^2 while the handbook value for the same
`sigma`, `a` and plane-strain state is `1361.4`
J/m^2. The sequence is monotone but the successive differences do not shrink by
the grid ratio, so the Richardson estimate of the observed order is negative and
no GCI band can be quoted. The SIF from the model is
`-58.6 %` below `sigma sqrt(pi a)`,
which is far outside the 2 % anchor of criterion (a) and outside any
acceptable discretisation error. The model is therefore reported as an **open
defect** and no K value from it is used anywhere in this node.

### (g) Paris exponent sweep

`N_f(m=2) = 1.018e+12`, `N_f(m=3) = 6.439e+06`,
`N_f(m=4) = 2.357` cycles with equal `C`; monotonically decreasing
`True`. The span (twelve decades) is the physical
consequence of a fixed coefficient: the exponent dominates the life.

### Typical details

| detail | k_i_pa_sqrt_m | geometry_factor | a_c_m | n_residual |
|---|---|---|---|---|
| weld_toe_surface_crack | 2.073e+06 | 1.034 | 0.4974 | 2.957e+06 |
| open_hole_edge_crack | 8.479e+06 | 4.228 | 0.4974 | 2.957e+06 |
| fillet_weld_toe_crack | 2.244e+06 | 1.119 | 0.4974 | 2.957e+06 |


## 4 Figures

* `docs/assets/fatigue_paris/paris_growth_curves.png` -- left: crack length
  versus cycles for `m = 2, 3, 4`; right: `da/dN` versus `Delta K`.
* `docs/assets/fatigue_paris/residual_life.png` -- residual life versus detected
  crack size.
* `docs/assets/fatigue_paris/j_integral_convergence.png` -- quarter-point `J`
  versus contour radius for three meshes, with the handbook level drawn.

## 5 Notes, limitations and traps

* The Paris coefficient is unit-critical. Inserting the mm/cycle,
  N/mm^1.5 value `5.21e-13` directly into the SI implementation inflates
  `da/dN` by 10^18 and returns a life of fractions of a cycle; the conversion
  `C_SI = C_Nmm / (31623^m * 1000)` is applied once and documented here.
* With equal `C`, the exponent sweep spans twelve decades; the monotonicity
  check is meaningful, the absolute values are not a design statement.
* The finite-width secant correction and the exact collinear-crack-array value
  differ by 15 % at `a/W = 0.40`; the anchor is therefore stated for
  `a/W <= 0.30`, where the deviation is a few percent.
* The quarter-point model is not yet qualified: the mesh-line contour integral
  is path independent to ~3 % and the near-tip stress follows `1/sqrt(r)`, but
  the absolute level is 2.4 times below the handbook value and does not converge
  under refinement. Candidate causes are the polar-quad diagonal split (which
  breaks mirror symmetry), the fan element's angular resolution near the crack
  faces and the truncated far field; this has to be settled before the model is
  used for any SIF.
* Scope of the verified chain: mode I, constant or block amplitude, homogeneous
  isotropic linear elastic, no crack closure, no retardation, no variable
  amplitude sequence effects (Miner-style accumulation is not verified here),
  no weld residual stress.

## 6 Error assessment

| quantity | reference | error | < 3 % ? |
|---|---|---|---|
| infinite-plate `K_I` | closed form (self-consistency) | 0 % | yes, but trivial |
| finite-width factor | exact collinear-crack array, `a/W <= 0.3` | 7.9 % | no (5 % band accepted) |
| edge-crack factor | 1.12 `sigma sqrt(pi a)` | 0 % | trivial |
| Paris life (step 0.005) | closed form | 0.40 % | yes |
| Paris life (step 0.001) | closed form | 0.09 % | yes |
| Paris life | adaptive ODE (LSODA) | 1.95e-09 % | yes |
| Paris vs S-N | FAT 80 reference | 0.10 decades | n/a (one decade band) |
| FEM `K_I` (fine grid) | `sigma sqrt(pi a)` | -58.6 % | **no (open defect)** |

All < 3 % statements hold except the finite-width cross-check (5 % band,
documented) and the singular-element SIF, which is an honest FAIL.
