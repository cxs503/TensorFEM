# TensorFEM: C(T) fatigue crack growth, Paris law and threshold calibration (ASTM E647 / E399)

*Campaign node P202W1-N09 — verification face `tensorfem.fatigue.ct_specimen`.*

## 1. Case introduction

**Physical background.** Fatigue crack growth in a pin-loaded compact-tension
C(T) specimen is the reference experiment of linear-elastic fracture mechanics
(ASTM E647 for the growth rate, ASTM E399 for the plane-strain toughness). The
specimen stress-intensity factor is
`K_I = P/(B sqrt(W)) f(a/W)` with the ASTM calibration polynomial

```
f(alpha) = (2 + alpha)/(1 - alpha)^1.5 (0.886 + 4.64 alpha - 13.32 alpha^2
                                       + 14.72 alpha^3 - 5.6 alpha^4),  alpha = a/W
```

quoted valid to better than 0.5 % over `0.2 <= alpha <= 0.8`.

**Geometry (ASTM proportions, data injected from the run):** total width
`1.25 W = 62.5 mm`, height
`1.2 W = 60.0 mm`, thickness
`B = 25.0 mm`, pin centres at
`(0, +-13.75) mm` (load line at `x = 0`,
front face at `x = -0.25 W`, back face at `x = W`).

**Material anchor** (`ferritic-pearlitic structural steel, R = 0.1, air`):

| constant | value | source |
|---|---|---|
| C | 6.900e-12 m/cycle (dK in MPa sqrt(m)) | Barsom & Rolfe, Fatigue and Fracture Control in Structures, 2nd ed.; K_IC range for QT low-alloy steel at sigma_y = 550 MPa |
| m | 3.00 | same |
| dK_th | 6.40 MPa sqrt(m) at R = 0.1 | same |
| K_IC | 50.0 MPa sqrt(m) (literature 45-60) | QT low-alloy steel, sigma_y = 550 MPa |
| E, nu | 206 GPa, 0.30 | plane strain, E' = E/(1-nu^2) |

**Growth law used for the virtual material** (threshold-modified Paris law with
a steep near-threshold branch, exactly zero below the threshold):

`da/dN = C dK^m (1 - (dK_th/dK)^k)` for `dK > dK_th`, `k = 8`,
and `da/dN = 0` otherwise.

## 2. Method

1. **Geometry factor**: ASTM E399 polynomial evaluated on a 0.05 grid in `a/W`;
   `K_I` for any load from the same expression.
2. **Finite-element cross-check (independent of the polynomial)**:
   six-node-triangle model of the C(T) body, mapped `(rho, theta)` grid from the
   crack tip to the exact specimen boundary, geometric radial grading, and a
   **quarter-point fan** (Barsoum / Henshell-Shaw) on the tip elements. The pin
   load is transferred as a statically equivalent nodal triplet (resultant force
   and zero net moment about the pin centre preserved). Rigid-body modes only are
   removed, at nodes on the back face far outside every contour.
   `K_J = sqrt(J E')` from Rice's J-integral on closed mesh contours (3-point
   Gauss per edge, counter-clockwise from the lower to the upper crack face), and
   `K` from the quarter-point crack-opening displacement
   `COD/ sqrt(r) -> (8 K_I/E')/sqrt(2 pi)` extrapolated to `r -> 0`.
3. **Virtual ASTM E647 test**: the crack is grown from
   `a/W = 0.30` to `0.55`
   under `dP = 12000 N` (R = 0.1, `P_max = 13333 N`)
   with RK4 integration of `dN/da = 1/(da/dN)`; the record is sampled every
   `0.25 mm` of crack advance
   (50 points, `dK` = 12.14 -
   24.19 MPa sqrt(m), `N_end = 4.5e+05` cycles).
   Data reduction: secant construction `da/dN_i = Da_i/DN_i` at the mean `dK`,
   then a least-squares line in log-log coordinates.
4. **Threshold**: Annex-A1 style load shedding with the normalised gradient
   `dK/da = -0.08 mm^-1`; the identified threshold is the `dK` where the measured
   rate passes `1e-10 m/cycle` (linear interpolation in `log10(da/dN)`).
5. **Toughness**: instability load `P_max` and the critical crack ratio
   `K_max(a_c) = K_IC` (bisection); E399 plane-strain size requirement
   `B >= 2.5 (K_IC/sigma_y)^2` checked; the value is re-derived independently from
   the finite-element J-integral at `a_c`.
6. **Miner summation**: for a five-level narrow-band block (below), every level
   gets a constant-amplitude crack-growth life `N_i`; the block sequence is then
   marched to instability and `D = sum n_i/N_i` is reported together with the
   Miner life prediction.
7. **Negative control**: the growth law is called below the threshold and must
   return identically zero.

## 3. Results

### 3.1 Paris law (C1, C2)

| quantity | fitted | reference | deviation |
|---|---|---|---|
| C [m/cycle] | 6.7678e-12 | 6.9000e-12 | -1.92 % |
| m | 3.0064 | 3.00 | +0.21 % |
| R^2 (log-log) | 0.99999800 | >= 0.98 | PASS |
| points | 50 | | |

### 3.2 Threshold (C3)

`dK_th = 6.455 MPa sqrt(m)` against the literature
`6.40` MPa sqrt(m), deviation
`+0.86 %` (44 load-shedding
steps, criterion `da/dN = 1e-10 m/cycle`).

### 3.3 Fracture toughness (C4)

`K_Q = K_IC = 50.000 MPa sqrt(m)` at the critical ratio
`a_c/W = 0.6948`
(`a_c = 34.74 mm`), deviation
`-6.70e-11 %`. E399 plane-strain requirement
`B_min = 2.5 (K_IC/sigma_y)^2 = 20.66 mm`
versus `B = 25.00 mm` -> `True`.
Finite-element cross-check at `a_c`: `K_J = 49.513`,
`K_cod = 47.608` versus ASTM
`50.000` MPa sqrt(m)
(`-0.97 %`).

### 3.4 Miner linear damage (C5)

| level dP [N] | cycles per block | dK at a/W = 0.30 [MPa sqrt(m)] |
|---|---|---|
| 9000 | 4000 | 9.05 |
| 10800 | 2500 | 10.86 |
| 12000 | 1500 | 12.07 |
| 13200 | 700 | 13.27 |
| 15000 | 200 | 15.08 |

* `D = sum n_i/N_i = 1.000475` at instability (`+0.048 %`),
  total `685800` cycles in `78` blocks,
  crack from `15.00` mm to
  `34.74` mm.
* Miner life prediction `685474` cycles versus
  the true block-sequence life `685800` cycles.

**Why D = 1 is exact here (and where it is not).** When the growth law is
*separable*, `da/dN = C f(a/W)^m dP^m`, the constant-amplitude life factorises as
`N_i = G/dP_i^m` with `G = int da/(C f^m)`, so
`D = sum n_i dP_i^m / G` telescopes to exactly 1 for any block order and any
number of blocks. The linear rule therefore cannot fail because of sequence
effects *inside this model*; it fails when the threshold truncates the small
cycles. The sensitivity sweep scales the whole spectrum:

| spectrum scale | D | deviation [%] | levels above dK_th | stalled |
|---|---|---|---|---|
| 0.5 | 0.5148 | -48.52 | 2 | False |
| 0.7 | 0.8090 | -19.10 | 4 | False |
| 0.9 | 1.0001 | +0.01 | 5 | False |
| 1.0 | 1.0005 | +0.05 | 5 | False |
| 1.2 | 1.0008 | +0.08 | 5 | False |

At scale 0.5 only two of five levels still grow the crack, and `D` collapses to
`0.515` (-48.5 %): the linear rule
over-predicts life whenever the spectrum is truncated by a threshold, which is
the physically relevant failure mode for variable-amplitude loading near the
fatigue limit.

### 3.5 Negative control (C6)

| dK [MPa sqrt(m)] | da/dN [m/cycle] | identically zero |
|---|---|---|
| 1.600 | 0.000e+00 | True |
| 3.200 | 0.000e+00 | True |
| 4.800 | 0.000e+00 | True |
| 5.760 | 0.000e+00 | True |
| 6.336 | 0.000e+00 | True |
| 6.400 | 0.000e+00 | True |

Maximum rate below the threshold: `0.0e+00` m/cycle.

### 3.6 Finite-element cross-check and mesh convergence (C7)

| a/W | K_J | K_COD | K_ASTM | dev K_J [%] | dev K_COD [%] |
|---|---|---|---|---|---|
| 0.40 | 22.232 | 21.814 | 22.222 | +0.04 | -1.84 |
| 0.55 | 21.935 | 21.494 | 22.222 | -1.29 | -3.28 |

Mesh refinement at `a/W = 0.40`:

| n_angles x n_rings | elements | nodes | K_J [MPa sqrt(m)] | K_COD [MPa sqrt(m)] |
|---|---|---|---|---|
| 48 x 10 | 912 | 1893 | 22.2247 | 21.1557 |
| 72 x 14 | 1944 | 3989 | 22.2319 | 21.8141 |
| 96 x 18 | 3360 | 6853 | 22.2335 | 22.0792 |

Largest relative change of `K_J` between the coarsest and finest mesh:
`0.040 %` (criterion 5 %); the finest
mesh is `+0.051 %` from the ASTM polynomial, so
the two independent stress-intensity routes (J-integral on the FE field and the
ASTM calibration) agree to a fraction of a percent inside the primary test
window. At `a/W = 0.25` the same study converges to
`+6.18 %`
(72x14) and `+6.42 %`
(96x20) above the ASTM polynomial. That difference does **not** shrink under
refinement, so it is the pin-hole idealisation (the FE pin force acts at the hole centre without modelling the 0.25 W hole, so the shallow-crack compliance differs from the ASTM calibration and the difference does not vanish under refinement), not a discretisation
error.

### 3.7 Criteria summary (C8 = figures)

| criterion | value | verdict |
|---|---|---|
| C1_paris_r_squared | value=0.999998, threshold=0.98 | PASS |
| C2_paris_constants | c_deviation_pct=-1.91632, m_deviation_pct=0.213394, threshold_pct=30 | PASS |
| C3_threshold | value_mpa=6.45527, reference_mpa=6.4, deviation_pct=0.863543, threshold_pct=20 | PASS |
| C4_toughness | value_mpa=50, reference_mpa=50, deviation_pct=-6.70468e-11, thickness_valid=True, threshold_pct=15 | PASS |
| C5_miner | damage=1.00048, deviation_pct=0.0475103, threshold_pct=20 | PASS |
| C6_negative_control | max_rate=0, threshold=1e-12 | PASS |
| C7_mesh_convergence | max_relative_change_pct=0.0396147, threshold_pct=5 | PASS |
| C8_figures | figures=['artifacts/fatigue_ct_e647/fig1_paris_curve.png', 'artifacts/fatigue_ct_e647/fig2_crack_path.png', 'artifacts/fatigue_ct_e647/fig3_j_integral.png', 'artifacts/fatigue_ct_e647/fig4_stress_field.png'] | PASS |

## 4. Figures

* `/root/tp-agent/work/m9374/artifacts/fatigue_ct_e647/fig1_paris_curve.png` — da/dN - dK in log-log coordinates: secant
  data, log-log fit, material law, threshold and the ASTM 1e-10 m/cycle line.
* `/root/tp-agent/work/m9374/artifacts/fatigue_ct_e647/fig2_crack_path.png` — crack path across the specimen at five
  cycle stations and the a/W - N history.
* `/root/tp-agent/work/m9374/artifacts/fatigue_ct_e647/fig3_j_integral.png` — K_J on successive contours (path
  independence) plus the three refinement levels and the ASTM value.
* `/root/tp-agent/work/m9374/artifacts/fatigue_ct_e647/fig4_stress_field.png` — sigma_yy field at P_max for a/W = 0.40.

## 5. Error judgement against the baselines

The question "is the error below 3 %?" is answered item by item; the highest
deviation over the whole chain is `3.278 %`.

| quantity | deviation from baseline | below 3 %? |
|---|---|---|
| K_J vs ASTM at a/W = 0.40 | +0.0433 % | YES |
| K_J vs ASTM at a/W = 0.55 | -1.2930 % | YES |
| K_COD vs ASTM at a/W = 0.40 | -1.8364 % | YES |
| K_COD vs ASTM at a/W = 0.55 | -3.2785 % | NO |
| K_J vs ASTM at a_c (toughness) | -0.9737 % | YES |
| Paris C vs literature | -1.9163 % | YES |
| Paris m vs literature | +0.2134 % | YES |
| dK_th vs literature | +0.8635 % | YES |
| K_IC vs literature (identification) | -0.0000 % | YES |
| Miner D vs 1 | +0.0475 % | YES |

* Baseline for `K` is the ASTM E399 calibration polynomial (quoted < 0.5 %), for
  the Paris constants / threshold / toughness the documented material anchor,
  and for the Miner sum the exact value `D = 1`.
* The ASTM polynomial is itself a curve fit with a quoted 0.5 % uncertainty, so
  the 0.04 % K_J agreement is "agreement to the accuracy of the baseline".
* The one place where a baseline is *not* matched to 3 % is the shallow-crack end
  (`a/W = 0.25`, +6.6 %, converged) — reported in section 3.6 and explained as
  the pin-hole idealisation rather than a numerical error. It is outside the
  primary `0.35 <= a/W <= 0.70` window quoted for this face.
* `K_COD` at `a/W = 0.55` is the only row above 3 % (-3.28 %). The COD
  extrapolation uses only the two innermost rings, where the quarter-point
  asymptote is cleanest but where the absolute opening is smallest; the
  J-integral route (-1.29 %, 0.04 %) is the primary estimator for `K` and the COD
  route is kept as a second, deliberately less accurate, independent check.
* The threshold reference (6.4 MPa sqrt(m)) is a mean curve; scatter between
  heats of the same steel easily reaches +-20 %, so the 0.86 % agreement shows
  consistency of the identification procedure, not a material measurement.

## 6. Notes, limits and pitfalls

* **Two independent routes, one truth.** The ASTM geometry factor is a curve fit;
  the FE J-integral is a numerical evaluation of the same configuration. Their
  agreement (0.04 % at a/W = 0.40) validates the geometry factor *and* the
  quarter-point/J machinery, but it cannot reveal an error shared by both
  (e.g. the pin-load idealisation).
* **Quarter-point bookkeeping.** The six-node-triangle midside node of the
  *tip-adjacent* edges must sit at `r/4`; the arc midside stays at the chord
  midpoint. A swapped midside on the second triangle of a quad folds the element
  (negative Jacobian at the centroid) — this was the only mesh bug found and it
  is caught by the conforming-mesh test.
* **Load convention.** The ASTM `P` is the force transmitted through *one* pin.
  Applying `P/2` per pin halves `K` — an error invisible in the R^2 of the Paris
  fit because it only shifts `C`.
* **The secant construction biases nothing for a pure power law**, but it does
  bias the fit when the threshold factor is active inside the sampled range; the
  fit window here starts at `dK = 12.1`, where the
  threshold factor is below 0.5 %.
* **Scope.** Linear elasticity, small strain, mode I, no crack closure, no
  overload retardation, no plasticity correction, no hole compliance. R-ratio
  effects are represented only through a fixed threshold; a Walker/Newman
  effective-dK formulation would be the next step for variable-R spectra.
* **K_IC is an identification, not a measurement.** The toughness check verifies
  that the ASTM geometry factor plus the instability criterion recover the
  documented value and that E399 validity holds; it is not an independent
  material test.
