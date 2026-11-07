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
* **Extended ladder** (7 rounds from a 12×12 mesh, `artifacts/adaptive_zz/study.json` →
  `extended_adaptive`): 170 → 1248 DOF with error 1.069 → 0.2455, measured rate **-0.691**,
  i.e. 2.8× the uniform rate and better than the `N^-1/2` ideal because the graded mesh
  resolves the singularity instead of fighting it.
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