# CASE-FEM-SCORDELIS-LO-ROOF -- Scordelis-Lo cylindrical roof

**Verdict: PASS** (11/11 gates). Primary anchor: metric-consistent cylindrical shell4 at 16x16 gives `w = -0.302469` vs the published `w_ref = -0.3024` = `+0.023 %` (gate 3 %). The historical flat-facet default (`sri`) reproduces its documented `+44.7 %` failure on the same mesh, while the same facets with assumed natural shear reach `-2.34 %`.

Reproduce: `PYTHONPATH=src python benchmarks/case_scordelis_lo_roof.py`  
Artifacts: `benchmarks/artifacts/CASE-FEM-SCORDELIS-LO-ROOF/case_result.{json,md}`  
Figure: `docs/assets/case-scordelis-lo-roof/scordelis_lo_roof_case.png`  
Tests: `PYTHONPATH=src python -m pytest tests/test_case_scordelis_lo_roof.py -q`

## 1. Problem statement

Open cylindrical roof with rigid end diaphragms under a uniform vertical surface load -- the standard test for membrane-bending coupling in curved shell elements. It is membrane dominated, so an element that locks in membrane/shear does not merely lose accuracy: it loses the load path.

| parameter | value |
|---|---|
| length_m | 50.0 |
| radius_m | 25.0 |
| half_angle_deg | 40.0 |
| thickness_m | 0.25 |
| young_pa | 432000000.0 |
| poisson | 0.0 |
| vertical_load_pa | 90.0 |
| support | rigid end diaphragms u_y = u_z = 0, one axial dof fixed |
| exact mid-surface area | 1745.3293 |
| reference `w_ref` | -0.3024 |

## 2. Method

No new element code. Three configurations run through their merged assembly paths on identical meshes (4x4, 8x8, 16x16):

- `cyl` -- metric-consistent cylindrical shell4 (curved geometry)
- `mitc` -- flat-facet Shell4, assumed natural transverse shear
- `sri` -- flat-facet Shell4, default one-point transverse shear

Audits attached to every solve, not assumed: (i) the assembled facet area against the analytic `L R 2*alpha` (a geometry budget, the shell analogue of validating `Cb/L/B` before trusting a resistance coefficient); (ii) the vertical reaction sum against the applied resultant `q * A_facet`; (iii) a rank census of one facet per shear rule (`tensorfem.shell4_rank.element_rank_census`), which splits the null space into rigid-body directions and genuine mechanisms and reports how many of the latter carry transverse displacement/rotation content.

Order and limit: observed order and the Richardson limit from the dyadic 4/8/16 ladder.

## 3. Results

### 3.1 Ladder (same facets, same load, same diaphragms)

| mesh | facet area deficit | sagitta / t | cylindrical `w` | err | flat+mitc `w` | err | flat+sri `w` | err |
|---|---|---|---|---|---|---|---|---|
| 4x4 | -0.5069 % | 1.519 | -0.216731 | -28.330 % | -0.357078 | +18.081 % | -0.492837 | +62.975 % |
| 8x8 | -0.1269 % | 0.381 | -0.296730 | -1.875 % | -0.284874 | -5.796 % | -0.405557 | +34.113 % |
| 16x16 | -0.0317 % | 0.095 | -0.302469 | +0.023 % | -0.295310 | -2.345 % | -0.437542 | +44.690 % |

### 3.2 Convergence

| configuration | observed order | Richardson limit | limit error |
|---|---|---|---|
| `cyl` | 3.801 | -0.302913 | +0.170 % |
| `mitc` | n/a | n/a | differences do not have a constant sign |
| `sri` | n/a | n/a | differences do not have a constant sign |

### 3.3 Rank census (one facet, chord 0.873)

| shear rule | zero modes | spurious | of which transverse | first elastic eigenvalue |
|---|---|---|---|---|
| `sri` | 9 | 3 | 3 | 327.20 |
| `mitc` | 7 | 1 | 0 | 327.20 |

### 3.4 Equilibrium

| configuration | worst `|sum(reaction_z) / (q A) - 1|` |
|---|---|
| flat facet (`sri`, `mitc`) | `1.07e-12` |
| cylindrical shell4 | `8.40e-03` |

The flat-facet family is exact to machine precision. The cylindrical element integrates the *exact* arc area and its reaction residual falls as `h^2`: it is exactly the rigid-mode projection `(K 1_z)^T u / (-1^T f)`, attributed to the interpolated rigid vertical translation in `docs/CASE-FEM-SCORDELIS-LO-ROOF-reaction-residual-attribution.md`, and is an order of magnitude below the displacement error it is compared with.

## 4. Figures

`scordelis_lo_roof_case.png`: (a) `|w|` vs elements per edge on a log-log axis with the reference line; (b) `|error|` at 16x16 against the 3 %% line; (c) the rank census (all spurious modes vs those with transverse content).

## 5. Notes and limits

- **The corrected claim.** The 50 % plateau is not a faceting limitation. The same faceted mesh passes once its transverse-shear block is rank sufficient, and the rank census says why: the one-point `sri` rule leaves 3 spurious mechanisms with transverse content, `mitc` leaves 0. The surviving `mitc` mechanism is an in-plane/drilling gauge direction that cannot store bending energy (see report 15618, `docs/` formulation-matrix study).
- **The default is not flipped here.** `plate_shear` still defaults to `sri`, because every existing flat-facet qualification would change. The case therefore pins the passing configuration explicitly and *inverts* the sri gate, so the documented failure cannot silently disappear.
- **Faceting is real but not the limiter on this benchmark.** At 16x16 the chord sagitta is 9.5 % of the thickness and the facet area is -0.032 % short of the analytic surface, yet the cylindrical and the `mitc` flat facet results agree with the reference to 0.02 % and 2.34 % respectively. What faceting costs here is bounded, not fatal.
- **Reference resolution.** `w_ref = -0.3024` is a four-significant-digit literature value (Scordelis & Lo 1960; MacNeal & Harder 1985); claims below ~0.1 %% are outside what it can certify, so the 16x16 cylindrical result is reported as agreement, not as a 0.02 %% validation.
- **Scope.** Linear statics, single load level, cylindrical geometry, thin-ish shell `R/t = 100`. Large deflection, buckling and general non-cylindrical curvature are outside this case.

## 6. Error determination

| quantity | value | reference | error | < 3 %? |
|---|---|---|---|---|
| `w`, cylindrical shell4, 8x8 | -0.296730 | -0.3024 | -1.875 % | **yes** |
| `w`, cylindrical shell4, 16x16 | -0.302469 | -0.3024 | +0.023 % | **yes** |
| `w`, flat facet + mitc, 16x16 | -0.295310 | -0.3024 | -2.345 % | **yes** |
| `w`, flat facet + sri (default), 16x16 | -0.437542 | -0.3024 | +44.690 % | **no** |
| vertical reaction / applied (flat facet, worst) | 1.000000000001068 | 1 | 1.1e-12 | yes |
| spurious transverse mechanisms, sri / mitc | 3 / 0 | 0 / 0 | exact integers | structural |

