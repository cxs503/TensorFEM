# SENT edge-crack J-integral in plane strain — ledger acceptance point

Anchor: `CASE-FEM-JINTEGRAL-EDGE-CRACK` (feature_ledger, status open at ticket
start).  Mission m17027.  This page closes the gap between the ledger
acceptance and the original anchor run:

* ledger acceptance: normalised `K_I / (sigma sqrt(pi a))` vs Tada's
  finite-width correction at **a/W = 0.3, plane strain**, tolerance < 5 %;
* the merged anchor (m16291, rechecked by m16362) graded **plane stress** at
  a/W = 0.20 / 0.35 / 0.50 only — the evidence page scope line reads
  "linear elastic plane stress";
* this page grades the same solver path with `plane="strain"`
  (`E' = E / (1 - nu^2)`) at a/W = 0.20 / 0.30 / 0.50 on the three-level
  ladder, and rebuilds the anchor artifact set (`result.json`,
  `gates_verified.json`, figures) that the m16362 audit flagged as missing
  from the repository (finding F1, disposition "request changes").

## 1. Case

| quantity | value |
|---|---|
| specimen | single-edge-notched tension (SENT) plate, upper half model |
| width W | 100 mm |
| crack ratios a/W | 0.20 / 0.30 / 0.50 (0.30 = ledger acceptance point) |
| height H/W | 2.0 (Tada fit quoted for H/W >= 1) |
| remote tension sigma | 100 MPa |
| E, nu | 210 GPa, 0.3 |
| constitutive plane | **plane strain** (`E' = E/(1-nu^2)`) |
| mesh | ray-fan T6 (quadratic triangle), quarter-point (Barsoum) singular ring at the tip |
| boundary | uniform tension on y = H, symmetry ahead of the crack, free crack face |

## 2. Method

Identical to the original anchor: Rice J-integral on the fan mesh lines
(three-point Gauss along the quadratic edges), full-plate `J = 2 x J_upper`,
`K_I = sqrt(J E')` with `E' = E/(1-nu^2)` in plane strain.  No handbook factor
enters the numerical path; Tada's polynomial

```
F(a/W) = 1.12 - 0.231 a/W + 10.55 (a/W)^2 - 21.72 (a/W)^3 + 30.39 (a/W)^4
```

is applied only in the grading step.  Reference: Tada, Paris & Irwin, *The
Stress Analysis of Cracks Handbook*, 3rd ed. (2000), quoted for a/W <= 0.6
and H/W >= 1 with +-0.5 % accuracy.

## 3. Results

| level | a/W | F_FEM | F_Tada | rel. error | n_elems |
|---|---|---|---|---|---|
| coarse | 0.20 | 1.3332 | 1.3707 | -2.73 % | 390 |
| coarse | 0.30 | 1.6271 | 1.6599 | -1.98 % | 390 |
| coarse | 0.50 | 2.7496 | 2.8264 | -2.72 % | 390 |
| medium | 0.20 | 1.3468 | 1.3707 | -1.74 % | 1050 |
| medium | 0.30 | 1.6376 | 1.6599 | -1.35 % | 1050 |
| medium | 0.50 | 2.7824 | 2.8264 | -1.56 % | 1050 |
| fine | 0.20 | 1.3563 | 1.3707 | -1.05 % | 2842 |
| fine | 0.30 | 1.6477 | 1.6599 | -0.74 % | 2842 |
| fine | 0.50 | 2.8027 | 2.8264 | -0.84 % | 2842 |

Acceptance point (a/W = 0.30, plane strain, finest mesh):

| K_I (FEM, finest) | 50583352.89 MPa·sqrt(m) |
| K_I (Tada) | 50959183.37 MPa·sqrt(m) |
| F_FEM / F_Tada | 1.6477 / 1.6599 |
| relative error | -0.74 % |
| ledger tolerance (< 5 %) | **PASS** |
| platform gate (< 3 %) | **PASS** |
| path spread (8 contours) | 0.16 % |
| reaction balance | 1.00e+00 |
| GCI (a/W=0.30) | not applicable: observed order 0.055 < 0.25 (flat sequence, constant H/W=2 bias); medium->fine change 0.61 % |

## 4. Figures

* `sent_plane_strain_convergence.png` — |K_FEM - K_Tada|/K_Tada over the
  ladder, with the 5 % ledger and 3 % platform gates;
* `sent_plane_strain_shape_factor.png` — recovered F(a/W) in plane strain
  against Tada's fit, with the plane-stress finest-mesh points of the
  original anchor run for comparison;
* `sent_plane_strain_path_independence.png` — F along the eight contour
  stations, finest mesh.

## 5. Notes / limitations

* linear elastic, small strain, stationary mode-I; no plasticity, no crack
  closure, no cohesive traction, no 3-D crack front;
* the strip height truncates the far field; the H sensitivity gate of the
  original anchor is **not_run** in this artifact set: the height stage fails
  deterministically with `CG did not converge: residual=0.0028` (rtol = 1e-12)
  at the H/W = 3.0 row of the sweep — a solver-robustness defect that needs
  its own ticket (cluster job cj1403 was submitted for the same stages);
* the residual error is one-signed (FE below the handbook) and decreases
  monotonically with h, consistent with the plane-stress behaviour;
* the non-singular control (mid-side node at the midpoint) reaches the same
  band at this mesh density, so the 3 % band is not a proof of the singular
  ring's necessity (honest limitation, inherited from the original anchor).

## 6. Error verdict

Key quantity ``K_I`` at the ledger acceptance point (a/W = 0.30, plane strain,
finest mesh) versus Tada's finite-width correction: **-0.74 %**,
i.e. **< 5 % (ledger) and < 3 % (platform): yes**.  All three plane-strain
cases are inside the 3 % band on the finest mesh (max 1.05 %).
Supporting gates: equilibrium 2.3e-13,
path spread < 0.43 %, GCI graded on the
asymptotic cases (max 2.01 %).
The `height` and `field` stages of the original anchor are recorded as
`not_run` in this artifact set (deterministic CG non-convergence at H/W = 3.0,
see §5); their gates are absent, not passed.
