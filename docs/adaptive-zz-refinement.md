# ZZ error estimation and indicator-driven h-refinement

Status: implemented and verified (`src/tensorfem/adaptive_zz.py`,
`tests/test_adaptive_zz.py`, `scripts/adaptive_zz_study.py`).  Reproduce with

```bash
PYTHONPATH=src python scripts/adaptive_zz_study.py     # artifacts + figures
PYTHONPATH=src python -m pytest tests/test_adaptive_zz.py -q
PYTHONPATH=src python scripts/adaptive_zz_report.py    # builds the mission report payload
```

Machine-readable evidence: `artifacts/adaptive_zz/study.json`,
`artifacts/adaptive_zz/summary.md`.  Figures: `docs/figures/adaptive_zz_*.png`.

## 1. Why the module exists

Before this change TensorFEM could *recover* a stress field at nodes
(`tensorfem.stress_recovery`, Barlow extrapolation plus area averaging) but could not
*estimate an error* and could not *refine a mesh adaptively*: every convergence study in
the repository is a uniform `h` ladder.  A solution whose error is concentrated in a small
part of the domain therefore had to be resolved by refining everything.

The missing face is added here as a self-contained, mesh-level capability:

| layer | API | what it does |
|---|---|---|
| recovery | `spr_recover`, `recover_at_gauss` | Zienkiewicz–Zhu superconvergent patch recovery: one least-squares linear polynomial per nodal patch fitted to the 2×2 Gauss-point stresses |
| estimation | `element_indicators`, `validate_indicators`, `effectivity_index` | `eta_e^2 = int_{Omega_e} (sigma_h - sigma*)^T C^-1 (sigma_h - sigma*) dOmega`, fail-closed validation |
| marking | `dorfler_mark` | Dörfler (bulk) marking: smallest element set carrying a requested fraction of the error energy |
| refinement | `refine_mesh`, `hanging_nodes`, `constraint_pairs` | quadtree split into four children, shared midpoints, chained hanging-node constraints flattened to multipoint constraints |
| loop | `adaptive_study`, `uniform_study`, `study`, `matched_dof_table`, `equal_error_table` | closed loop, uniform reference ladder, equal-DOF and equal-accuracy comparisons |

## 2. Verification problem: re-entrant corner (Williams) singularity

Domain `Omega = [-1,1]^2 \ [-1,0]x[-1,0]` (a square with the third quadrant removed), so
the interior angle at the corner is `2*alpha = 3*pi/2`.  The symmetric eigen-solution of the
homogeneous plane problem is the biharmonic Airy function

```
phi(r, theta') = r^(lambda+1) [ A cos((lambda+1) theta') + B cos((lambda-1) theta') ],
theta' = theta - pi/4,   sin(2 alpha lambda) + lambda sin(2 alpha) = 0  ->  lambda = 0.5444837367825
```

with `B` fixed by the traction-free condition on both notch faces.  The field has zero body
force, zero traction on the notch faces and a `sigma ~ r^(lambda-1) = r^-0.455` stress
singularity.  Prescribing its trace on the outer square as Dirichlet data defines a boundary
value problem whose exact solution *is* that field, which gives three independent reference
quantities:

* exact stress and strain at every point (`williams_stress`, `williams_strain`),
* exact displacement (`williams_displacement`, validated against the strain–stress relation),
* exact energy norm `a(u,u) = int sigma : C^-1 : sigma dOmega` in closed form
  (`exact_energy_norm_sq`; the radial power law is integrated analytically, only the smooth
  angular integral is quadrature).

This is the canonical model of a ship-structure detail such as a hatch-coaming corner or a
bracket toe, and it is the classical benchmark for ZZ adaptivity because the uniform
convergence rate is limited by the singularity: `||u-u_h||_E ~ N^(-lambda/2) = N^-0.272`.

Physical parameters (dimensionless): `E = 1`, `nu = 0.3`, plane strain, unit thickness,
amplitude `A = 1`, half extent `H = 1`.

## 3. Method

1. **Model problem.** Q4 plane-strain elasticity, exact Dirichlet data on the outer square,
   natural (traction-free) notch faces, own vectorised assembly (`element_stiffness`,
   `assemble`) cross-checked against `tensorfem.continuum.q4_stiffness` (agreement 2e-16).
2. **Hanging nodes.** Refinement is non-conforming; every node sitting at the midpoint of an
   element edge is eliminated by `u_hang = (u_a + u_b)/2` per component, which is exactly the
   Q4 restriction on a straight edge.  Chained constraints (a hanging node on the edge of an
   element that is itself hanging) are flattened transitively, and prescribed parents are
   folded into the MPC offset.  The constrained space is conforming, which
   `test_mesh_with_hanging_nodes_passes_the_linear_patch_test` proves by passing the linear
   patch test on a refined mesh.
3. **Indicator.** `sigma_h` is sampled at the 2×2 Gauss points (superconvergent points),
   `sigma*` is the SPR-recovered nodal field re-interpolated to the same points, and the
   energy-norm difference is integrated with the 2×2 rule.
4. **Loop.** solve → recover → indicate → Dörfler mark (`fraction = 0.5`) → refine, repeated
   for two rounds from a 16×16 start mesh (192 elements).
5. **Independent truth.** The true error is *not* taken from the estimator: it is computed by
   graded Gauss quadrature of `(sigma_h - sigma_exact) : C^-1 : (sigma_h - sigma_exact)` with
   the number of subcells graded towards the corner (`true_element_error_sq`).  Quadrature
   convergence is measured: the value changes by 3.0e-3 relative between 4 and 16 subcells and
   is unchanged (to round-off) from 8 subcells upwards.

## 4. Results

Convergence and effectivity (`artifacts/adaptive_zz/study.json`):

| mesh | DOF | true `||u-u_h||_E` | estimated `||eta||` | effectivity |
|---|---|---|---|---|---|
| uniform 8×8 | 66 | 1.3262 | 1.3077 | 0.986 |
| uniform 16×16 | 322 | 0.9167 | 0.9099 | 0.993 |
| uniform 24×24 | 770 | 0.7378 | 0.7330 | 0.994 |
| uniform 32×32 | 1410 | 0.6321 | 0.6282 | 0.994 |
| uniform 48×48 | 3266 | 0.5079 | 0.5050 | 0.994 |
| uniform 64×64 | 5890 | 0.4348 | 0.4323 | 0.994 |
| adaptive start (16×16) | 322 | 0.9167 | 0.9099 | 0.993 |
| adaptive round 1 | 332 | 0.7340 | 0.7511 | 1.023 |
| adaptive round 2 | 354 | 0.6767 | 0.6988 | 1.033 |

* The estimator is **asymptotically exact**: `theta = 0.986 -> 0.994` on the uniform ladder and
  `1.03` on the graded mesh (a constant over-estimate of a few percent, which is harmless for
  marking).  Estimated and true errors decrease together, so the recovery is not merely small.
* **Uniform rate** `d log e / d log N = -0.249` versus the theoretical `-lambda/2 = -0.272`;
  the two bracketing estimates converge towards the theory as `h -> 0`.
* **Two adaptive rounds**: 354 DOF give `0.6767` where the cheapest uniform mesh with at least
  as many unknowns (770 DOF = 2.18×) gives `0.7378`, i.e. **9% lower error at 2.2× less work**.
  Interpolating the uniform ladder in log–log, `0.6767` costs **992 uniform DOF**: adaptivity
  saves **2.80×** DOF at equal accuracy.
* **Extended ladder** (11 rounds from a 12×12 mesh, `artifacts/adaptive_zz/study.json` →
  `extended_adaptive`): 170 → 7690 DOF with error 1.069 → 0.0978.  A single least-squares
  slope over the whole ladder gives **-0.577**, which is *steeper than the optimal-mesh bound*
  `N^-1/2` and is therefore **not** a rate: the first two rounds collapse a boundary-layer
  error at local rates of `-3.9` and `-1.6` and that pre-asymptotic part contaminates a global
  fit.  The quantity comparable with theory is the tail fit: **-0.4933** over the last six
  intervals against the bound `-0.5` (**1.3 %**), with per-interval rates
  `-0.523, -0.501, -0.441, -0.561, -0.500, -0.439, -0.531` oscillating about it.  The
  overstatement is now detected automatically (`criteria.global_rate_overstates_theory =
  true`, `adaptive_zz.asymptotic_summary`) and asserted in
  `test_adaptive_rate_reaches_the_best_approximation_bound`.
* **Localisation**: the Dörfler set is tiny and sits at the corner — round 1 marks 3 elements
  (1.6% of the mesh) carrying 54% of the estimated error energy at a maximum distance of
  0.088 from the corner; round 2 marks 5 elements (2.5%) carrying 51%.
* **Recovered stress vs the exact (higher-order) reference** in the analytic region
  (`x <= -0.3, y >= 0.3`): relative nodal error `6.8e-2 -> 3.6e-3` over the uniform ladder,
  i.e. converging.  Near the corner the recovered field *cannot* converge at that rate (the
  exact stress is unbounded there), which is why the effectivity is reported globally and the
  recovery error is reported on both the whole domain and the smooth sub-region.

## 5. Pitfalls recorded during development

1. **`numpy.linalg.inv/solve` is pathologically slow on this platform** (milliseconds for a
   3×3 matrix, seconds per estimator evaluation at 48 elements).  The 2×2 and 3×3 inverses are
   now closed-form (`inverse3`, `_strain_matrix`), which took the full study from 50 s per
   solve to 0.13 s.
2. **A topological boundary test is wrong for non-conforming meshes.**  Counting edges used by
   one element classifies every hanging node (and its parents) as a boundary node, which then
   collides with the Dirichlet set.  The boundary is now detected geometrically for the known
   domain, and hanging nodes are provably interior (a boundary edge is subdivided only when its
   own element is refined).
3. **`B u` is the strain, not the stress.**  An earlier revision fed `B u` into the indicator
   and the recovered field; the effectivity was then a meaningless constant `1.86` and the
   recovered stress appeared not to converge (71% error in a *smooth* region).  The fix
   (`C : B u`) moved the effectivity to `0.99` and the smooth-region recovery error to `3.6e-3`.
4. **A 270° domain has two planar orientations.**  Removing the *fourth* quadrant and using the
   eigen-solution of the *third* quadrant produces fully consistent strains but an inconsistent
   boundary value problem: the FE solution then converges to a wrong field (interior
   displacement error growing with refinement).  The notch must be the third quadrant because
   the bisector of the material angle is then at `theta = pi/4`, the angle used by the angular
   eigenfunctions.
5. **`||u||^2 - ||u_h||^2` is not the error norm for a Dirichlet-driven problem.**  With
   prescribed boundary data the discrete solution does not live in the test space, so the
   Galerkin identity carries a lifting term.  The true error is therefore measured by direct
   quadrature, never by the energy difference.

## 6. Limitations

* Linear elasticity, small strain, single material, plane stress or plane strain, Q4 elements
  only.  No shells, plates, contact, plasticity or nonlinear path is exercised by the loop.
* The indicator is energy-norm ZZ, not goal-oriented (no dual-weighted residual, no
  quantity-of-interest control), and it does not distinguish discretisation from modelling
  error.
* Refinement is isotropic quadtree subdivision with hanging-node elimination; no anisotropic
  or `p`-refinement, no coarsening, no mesh smoothing.
* The marking fraction is fixed (`0.5`); no optimality proof of the Dörfler set is claimed, and
  the measured effectivity is evaluated, not assumed.
* The two-round adaptive gain (2.8× DOF at equal accuracy) is a *short-horizon* result on one
  singular problem; the asymptotic statement is the extended ladder's rate, which is a
  single-problem measurement and not a general performance claim.
## 7. Continuation — orders against closed-form theory, and a cost baseline

Two follow-up modules close the accounting gap left by the asymptotic bracket used above
(`-lambda/2` is a bracket, and the true error in this problem is only available by quadrature):

**Correction (2026-11-11 audit).** An earlier revision of this page quoted the global slope
`-0.691` as "2.8× the uniform rate and better than the `N^-1/2` ideal".  That statement was
wrong in kind, not in arithmetic: an adaptive method cannot beat the best-approximation bound
`N^-1/2` (Binev–Dahmen–DeVore 2004; Stevenson 2007), so a steeper measured slope can only be
pre-asymptotic.  The corrected accounting lives in
`artifacts/adaptive_zz/benchmark_table.md` (row 3) and is enforced by a test.

* `tensorfem.adaptive_1d` — P1 h-adaptive loop on two analytic 1D problems whose **discrete
  energy error is a closed form** (`u = sin(pi x)` and `u = x^0.6`). Measured uniform orders:
  `-0.9722` vs theory `-1` (2.8 %), `-0.0977` vs theory `-(alpha-1/2) = -0.1` (2.3 %). The
  adaptive loop reaches `-0.5027` and, at equal accuracy on the singular problem, needs **>= 39x**
  fewer unknowns than the uniform ladder; on the smooth problem it is the *control* case and
  costs 2x more.
* `tensorfem.adaptive_poisson_2d` — Q4 scalar Poisson with a polynomial bubble manufactured
  solution and exact quadrature: energy order `-0.4848` and L2 order `-0.9704` against theory
  `-0.5` / `-1.0` (3.0 % each at the finest rung pair), and a recovered-gradient indicator whose
  effectivity reaches `theta = 1.0014` at 1521 unknowns. That value is the missing piece for this
  page: the few-percent effectivity bias measured here is a property of the **singular field**,
  not of the estimator family, because the same estimator is asymptotically exact on a smooth one.

Details, figures, cost baselines and the error-by-error judgement (including the rows that fail
the 3 % gate) are in `docs/adaptive-continuation-order-cost.md`; the generated payloads are
`artifacts/adaptive_continuation/*.json` + `tables.md`.

## 8. Asymptotic order accounting and the acceptance criteria (first round of the GAP sweep)

The mission criterion is two-part: (i) the refinement sequence must reproduce the theoretical
convergence order, and (ii) the ZZ efficiency index must lie in `theta in [0.8, 1.2]`.  Both
are now machine-checked rather than asserted, by `adaptive_zz.local_rates`,
`adaptive_zz.tail_rate` and `adaptive_zz.asymptotic_summary`, and emitted by
`scripts/adaptive_evidence_pack.py` into `artifacts/adaptive_zz/benchmark_table.md` /
`evidence.json` (every number read from the study JSON, never retyped; the manifest carries
the sha256 of each payload plus the commit).

### 8.1 L-shape order table (adaptive ladder, 11 rounds)

| round | DOF | `||u-u_h||_E` | local rate | effectivity |
|---|---|---|---|---|
| start 12×12 | 170 | 1.0688 | — | 0.991 |
| 1 | 180 | 0.8530 | -3.946 | 1.023 |
| 2 | 202 | 0.7851 | -0.720 | 1.033 |
| 3 | 248 | 0.5679 | -1.578 | 1.047 |
| 4 | 332 | 0.4670 | -0.671 | 1.029 |
| 5 | 486 | 0.3826 | -0.523 | 1.019 |
| 6 | 772 | 0.3035 | -0.501 | 1.013 |
| 7 | 1248 | 0.2455 | -0.441 | 0.999 |
| 8 | 1940 | 0.1917 | -0.561 | 1.005 |
| 9 | 3100 | 0.1516 | -0.500 | 1.004 |
| 10 | 4992 | 0.1230 | -0.439 | 0.999 |
| 11 | 7690 | 0.0978 | -0.531 | 0.996 |

### 8.2 Criterion verdicts

| criterion | reference | measured | rel. error | verdict |
|---|---|---|---|---|
| Williams `lambda` | 0.5444837367825 (Williams 1952) | 0.5444837367825 | 0.00 % | PASS |
| uniform order, L-shape (tail, 3 intervals) | `-lambda/2 = -0.2722` | -0.2599 | 4.5 % | PASS (< 15 %) |
| adaptive order, L-shape (tail, 6 intervals) | `-1/2` (optimal-mesh bound) | -0.4933 | 1.3 % | PASS (< 15 %) |
| ZZ effectivity, 21 L-shape levels | `theta in [0.8, 1.2]` | 0.986 – 1.047 | worst 4.7 % of 1 | PASS |
| 2D Poisson order, energy / L2 | -0.5 / -1.0 | -0.4848 / -0.9704 | 3.0 % / 3.0 % | PASS |
| 2D Poisson effectivity (1521 unknowns) | 1 | 1.00142 | 0.14 % | PASS |
| 1D P1 order, smooth / singular | -1 / -0.1 | -0.9722 / -0.0977 | 2.8 % / 2.3 % | PASS |
| 1D P1 singular effectivity | 1 | 0.2329 | 77 % | **FAIL — documented** |

The last row is kept as an honest FAIL: the 1D gradient-recovery estimator is a stable
*underestimate* on the `x^0.6` field, so the `theta in [0.8, 1.2]` interval is asserted only
for the 2D indicators, which meet it.  Claiming otherwise would mean rescaling the estimator
after seeing the answer, which is exactly what §6 forbids.

### 8.3 What is *not* established

* The `-1/2` tail is a single-problem measurement on one singularity strength (`lambda =
  0.544`); no claim is made about other exponents or about 3D.
* The equal-accuracy saving (2.80× DOF) is a two-round, short-horizon number.
* Capability statement: this closes *linear-elastic Q4 ZZ estimation + isotropic h-adaptivity
  with hanging-node elimination*, verified on an L-shape and on a smooth manufactured field.
  It does not close goal-oriented adaptation, anisotropic refinement, `p`-adaptation or
  nonlinear/contact paths.
