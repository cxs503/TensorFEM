# CASE-FEM-SCORDELIS-LO-ROOF — the reaction residual, attributed

**Verdict: open item closed** (10/10 gates). The metric-consistent cylindrical
shell4 misses the vertical reaction resultant by `8.404e-03` at 4x4 and
`5.347e-04` at 16x16, which the merged case report
(`docs/CASE-FEM-SCORDELIS-LO-ROOF.md`, gate `equilibrium.cylindrical_reaction`)
explicitly left "reported rather than explained (open item)".  It is now named,
derived and measured: the element does not annihilate the *interpolated* rigid
vertical translation, because the tangential component of that translation varies
as `sin(theta)` and a bilinear facet reproduces it only to `O(pitch^2)`.  The
residual is a curved-metric discretisation limit, **not** a load, reaction-extraction
or ring-indexing defect, and it does not touch the 3 % displacement verdict.

Reproduce: `PYTHONPATH=src:benchmarks/verified python benchmarks/verified/fem_shell_scordelis_lo/reaction_residual_study.py`
Artifacts: `benchmarks/verified/fem_shell_scordelis_lo/reaction_residual_study.json`
Figure: `benchmarks/verified/fem_shell_scordelis_lo/reaction_residual_study.png`
Tests: `PYTHONPATH=src python -m pytest tests/test_cylindrical_shell4_rigid_translation.py -q`

## 1. Case and the question

Scordelis-Lo roof, unchanged from the merged case: `L = 50`, `R = 25`,
half angle `40 deg`, `t = 0.25`, `E = 4.32e8`, `nu = 0`, vertical surface load
`q = 90`, rigid end diaphragms (`u_y = u_z = 0`) plus one axial dof, reference
mid-edge deflection `w_ref = -0.3024`.

| quantity | value |
|---|---|
| loaded surface area (exact) | 1745.3293 |
| applied vertical resultant `q A` | 157079.6327 |
| finest-mesh `w` error | +0.0228 % |
| flat-facet residual (control, 4x4) | -5.362e-14 |

The question is narrow: *which term* produces a reaction residual on the curved
element while the flat-facet family is exact to machine precision?

## 2. Method

**Exact algebraic identity.** For a linear static solve with homogeneous essential
boundary conditions, `r = K u - f` with `r = 0` on the free dofs, so for any
assembled `K` and the unit global z-translation `1_z`:

    sum_z(r) - (-1^T f) = 1_z^T K u = (K 1_z)^T u .

Therefore the residual is nonzero *if and only if* `K 1_z != 0`, i.e. if and only
if the element does not annihilate the interpolated rigid vertical translation. No
hypothesis about load lumping or reaction extraction is needed: the identity is
checked numerically on the same `K` the solver used (`ShellBenchmarkResult.stiffness`,
newly exposed - additive field, no behaviour change).

**Closed form for the spurious strain.** For a unit vertical translation of one
facet the local fields are `v = -sin(theta)`, `w = cos(theta)` with `s = R theta`.
Bilinear interpolation at the centroid is off by `-(h^2/8) f''` and the interpolated
tangent derivative by `-(h^2/24) f'''`, hence

    eps_s    = v,s + w/R  = h^2 cos(theta_c) / (12 R^3)
    gamma_sz = w,s - v/R  = h^2 sin(theta_c) / (12 R^3)

with `h` the facet pitch in arc length.  Predicted RMS strain of the assembled mesh
is the area-weighted mean of `eps^2/(1-nu^2) + (5/12) gamma^2/(1+nu)` over facet
centres, since `1_z^T K 1_z = sum_e 1_z^T K_e 1_z`.

**Experiments.** (i) 4/8/16 mesh ladder of the unmodified benchmark, each solve
audited by the identity above; (ii) single-facet pitch ladder at fixed `R = 25`;
(iii) single-facet radius ladder at fixed pitch; (iv) flat-facet control on the
same problem; (v) axial-translation control (`K 1_x`, which has no `sin(theta)`
modulation and must be exact); (vi) a deliberate round-off probe: the energy of a
facet is a difference of `O(E t h)` stiffness entries, so every point carries a
signal-to-noise ratio `1_z^T K 1_z / (eps * sum|K_ij|)` and points below `SNR 100`
are reported as floor-limited instead of being fitted as physics
(probe at pitch `0.1091`: SNR `0.18` -> excluded).

## 3. Results

### 3.1 Mesh ladder: the residual is the rigid-mode projection

| mesh | dofs | applied | reaction | reaction/applied - 1 | \|(K 1_z)^T u / (-1^T f)\| | identity gap | spurious RMS strain | closed form |
|---|---|---|---|---|---|---|---|
| 4x4 | 150 | 1.570796e+05 | 1.583997e+05 | 8.403829e-03 | 8.404e-03 | 2.0779e-13 | 3.880248e-04 | 3.892e-04 |
| 8x8 | 486 | 1.570796e+05 | 1.574140e+05 | 2.128559e-03 | 2.129e-03 | 1.5876e-13 | 9.705920e-05 | 9.713e-05 |
| 16x16 | 1734 | 1.570796e+05 | 1.571636e+05 | 5.346674e-04 | 5.347e-04 | 3.5473e-13 | 2.426822e-05 | 2.427e-05 |

The identity closes to `3.547e-13` (machine precision, relative to a
`1.57e+05` load): the residual has exactly one source.  Its observed order is
`1.987` — the interpolation-error rate.

### 3.2 The spurious strain is predicted, not fitted

| mesh | closed-form RMS strain | measured / predicted |
|---|---|---|
| 4x4 | 3.892e-04 | 0.997 |
| 8x8 | 9.713e-05 | 0.999 |
| 16x16 | 2.427e-05 | 1.000 |

### 3.3 Single facet: pitch and curvature scaling

| pitch [m] | energy density [N/m] | RMS strain | measured/predicted | energy SNR | round-off limited |
|---|---|---|---|---|---|
| 0.4363 | 5.3672e-05 | 9.9696e-07 | 0.999857 | 470.9 | no |
| 0.8727 | 8.5895e-04 | 3.9883e-06 | 0.999973 | 30139.8 | no |
| 1.7453 | 1.3741e-02 | 1.5952e-05 | 0.999878 | 1926820.3 | no |
| 3.4907 | 2.1969e-01 | 6.3783e-05 | 0.999513 | 122778096.3 | no |

Log-log slope of the energy density against pitch: `4.000` (expect 4, i.e. strain ~ pitch^2).

| R [m] | energy density [N/m] | RMS strain | measured/predicted | energy SNR | round-off limited |
|---|---|---|---|---|---|
| 25 | 1.3741e-02 | 1.5952e-05 | 0.999878 | 1926820.3 | no |
| 50 | 2.1474e-04 | 1.9941e-06 | 0.999973 | 30139.8 | no |
| 100 | 3.3545e-06 | 2.4924e-07 | 0.999857 | 470.9 | no |
| 200 | 4.9873e-08 | 3.0390e-08 | 0.975321 | 7.0 | yes |

Log-log slope against radius at fixed pitch: `-6.000` (expect -6, i.e. strain ~ 1/R^3).
At the facet centre `theta_c = 0` the energy density is `8.9076e-04` N/m, i.e. the
membrane branch `h^2/(12R^3)` with no shear contribution — the `sin`/`cos` split of the
closed form is visible in the data.

### 3.4 Controls

| control | result |
|---|---|
| flat-facet Shell4 (`mitc`) 4x4 / 8x8, same problem | `-5.362e-14` / `5.611e-13` |
| axial translation `\|K 1_x\| / |K|_max` (finest) | `1.128e-15` |
| 16x16 displacement error (unchanged verdict) | `+0.0228 %` |

The flat-facet element reproduces a rigid translation exactly on the same facets,
nodes and load, so the residual cannot come from the load integral or from the
reaction extraction — both are shared code paths.

### 3.5 Gates

| gate | quantity | value | bound | verdict |
|---|---|---|---|---|
| `algebra.rigidmode_identity` | max |residual - (K 1_z)^T u / (-1^T f)| over 4/8/16 | 3.54731e-13 | < 1e-10 (relative to the applied load) | PASS |
| `attribution.spurious_rigid_translation` | min over ladder of spurious RMS strain under unit vertical translation | 2.42682e-05 | > 0, i.e. K 1_z != 0 | PASS |
| `attribution.residual_order` | observed order of |residual| over 4/8/16 | 1.98717 | >= 1.8 (h^2 from bilinear interpolation error) | PASS |
| `attribution.pitch_scaling` | log-log slope of spurious energy density vs facet pitch at R = 25 | 3.99967 | |slope - 4| <= 0.4 (energy ~ strain^2 ~ pitch^4 at fixed R) | PASS |
| `attribution.curvature_scaling` | log-log slope of spurious energy density vs radius at fixed pitch | -6.00003 | |slope + 6| <= 0.6 (strain ~ pitch^2 / R^3) | PASS |
| `attribution.axial_translation_exact` | max over ladder of |K 1_x| / |K|_max | 1.12816e-15 | < 1e-12 | PASS |
| `control.flat_facet_exact` | max |reaction/applied - 1|, flat-facet Shell4 on the same problem | 5.61107e-13 | < 1e-9 | PASS |
| `attribution.prediction_facet` | max |measured / derived spurious strain - 1| over the single-facet pitch sweep | 0.000487303 | <= 5 % (derivation drops only O(dtheta^2) terms) | PASS |
| `attribution.prediction_mesh` | max |measured / derived mesh spurious strain - 1| over 4/8/16 | 0.00304286 | <= 5 % | PASS |
| `accuracy.unchanged` | w error of the cylindrical element at 16x16 | 0.000228344 | |error| < 3 % | PASS |

## 4. Figure

`reaction_residual_study.png`: (a) `|residual|` and `|(K 1_z)^T u / (-1^T f)|` on a
log-log axis against an `h^2` reference; (b) single-facet energy density against
facet pitch with a `pitch^4` reference; (c) energy density against cylinder radius
with an `R^-6` reference, floor-limited radii marked and excluded from the fit.

## 5. Notes and limits

- **Nothing was changed to make the residual disappear.** The element, the default
  shear rule and every published number stay as they were; the merged case keeps
  its `equilibrium.cylindrical_reaction` gate (`< 1 %`).  This report converts an
  "unexplained" into a derived limit and adds an attribution gate to the anchor.
- **Scope.** Linear statics, `nu = 0`, one thickness (`t/R = 0.01`), selective
  one-point membrane/shear integration (the element's shipped rule).  The
  derivation uses the leading `O(h^2)` interpolation term; at `4 deg` facets the
  measured/predicted ratio is still `1.0000` (see 3.3), so the truncated terms are
  small over the whole tested range.
- **Round-off wall.** Below roughly `pitch = 0.22` m the single-facet energy falls
  under `eps * sum|K_ij|`; such points are reported with their SNR and excluded
  from fits rather than being interpreted.  The mesh-level numbers (3.1, 3.2) are
  unaffected because they are 2-4 orders larger.
- **What this does *not* say.** It does not say the cylindrical element is
  inaccurate: at 16x16 its displacement error is `+0.0228 %`, and the residual is
  `2.34x` *smaller* than that error.  It also does not re-open the
  flat-facet root cause (report `docs/CASE-FEM-SCORDELIS-LO-shear-rank-diagnosis.md`).
- **Actionable hint.** Any future "curved element passes but reaction is off"
  observation should be screened with the identity in section 2 before being filed
  as a solver bug; `K 1_r` for each rigid translation is a two-line audit.
- Environment: `loongarch64`, torch 2.7.1 / Python 3.11.6, `torch.float64`, dense
  LU.  Wall clock of the whole study `19.8 s`.

## 6. Error judgement against reference values

| key quantity | reference | measured | error | < 3 % ? |
|---|---|---|---|---|
| mid-edge `w`, 16x16 | `-0.3024` (Scordelis & Lo) | `-0.302469` | `+0.0228 %` | **yes** |
| residual order | 2 (bilinear interpolation) | `1.987` | `0.64 %` | **yes** |
| pitch scaling exponent | 4 | `4.000` | `0.01 %` | **yes** |
| curvature scaling exponent | -6 | `-6.000` | `0.00 %` | **yes** |
| spurious strain, mesh ladder | closed form (this report) | see 3.2 | max `0.304 %` | **yes** |
| spurious strain, single facet | closed form (this report) | see 3.3 | max `0.0487 %` | **yes** |
| reaction residual | `(K 1_z)^T u / (-1^T f)` | ladder | `3.547e-13` (abs) | **yes** |
| flat-facet residual (control) | 0 | `5.611e-13` | n/a (machine precision) | **yes** |

No benchmark tolerance was relaxed anywhere.  The single honest caveat: the
*closed form itself* is derived in this report, so its agreement is a consistency
check between derivation and code, not an independent experimental measurement.
