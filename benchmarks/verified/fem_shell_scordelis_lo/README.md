# Scordelis-Lo Cylindrical Roof Benchmark

## Overview

The Scordelis-Lo cylindrical roof is one of the most classic benchmarks for shell finite elements, testing the ability to capture membrane-bending coupling in curved shells. This verification compares two formulations available in TensorFEM:

1. **Shell4 (flat facet)**: Standard Q4 membrane + Mindlin plate element using flat facets to approximate curved geometry
2. **Cylindrical shell4**: Specialized element with exact cylindrical metric and curvature terms

## Problem Definition

| Parameter | Value |
|-----------|-------|
| Geometry | Cylindrical roof |
| Length L | 50 |
| Radius R | 25 |
| Half-angle | 40° |
| Thickness t | 0.25 |
| Young's modulus E | 4.32×10⁸ |
| Poisson's ratio ν | 0 |
| Load | Vertical surface load q = 90 |
| Boundary conditions | Rigid diaphragms at curved ends (u_y = u_z = 0) |

**Reference solution**: Vertical displacement at midpoint of free edge = **-0.3024**

Source: Scordelis & Lo (1960), widely cited in FEM literature.

## Results

### Flat Facet Shell4

| Mesh | Elements | Nodes | Displacement | Error |
|------|----------|-------|--------------|-------|
| 4×4 | 16 | 25 | -0.4928 | 62.98% |
| 8×8 | 64 | 81 | -0.4056 | 34.11% |
| 16×16 | 256 | 289 | -0.4375 | 44.69% |
| 32×32 | 1024 | 1089 | -0.4518 | 49.40% |

**Verdict**: ❌ **FAIL**
- Error exceeds 3% threshold on all meshes
- Non-monotone convergence (error increases with refinement)
- Root cause: Flat facet approximation cannot capture membrane-bending coupling in curved shells

### Cylindrical Shell4

| Mesh | Elements | Nodes | Displacement | Error |
|------|----------|-------|--------------|-------|
| 4×4 | 16 | 25 | -0.2167 | 28.33% |
| 8×8 | 64 | 81 | -0.2967 | 1.88% |
| 16×16 | 256 | 289 | -0.3025 | 0.02% |
| 32×32 | 1024 | 1089 | -0.3037 | 0.41% |

**Verdict**: ✅ **PASS**
- Error within 3% threshold on meshes ≥ 8×8
- Monotone convergence with mesh refinement
- Best result: 0.02% error at 16×16 mesh

## Update (m16449): the flat-facet failure is a rank deficiency, not faceting

The two tables above are the *default-rule* record and stay reproducible
(`result.json`, `gates_verified.json` are unchanged).  The **interpretation**
below them ("flat facets cannot capture membrane-bending coupling") has since
been tested and **falsified** -- see
`docs/CASE-FEM-SCORDELIS-LO-shear-rank-diagnosis.md` and reproduce with

```bash
OMP_NUM_THREADS=8 python3 shear_treatment_study.py   # writes shear_treatment_study.{json,png}
```

Holding nodes, facets, load and diaphragms fixed and changing **only** the
transverse-shear integration rule of the same flat-facet element:

| rule | spurious zero-energy modes (24x24 facet) | 16x16 error | 32x32 error |
|---|---:|---:|---:|
| `sri` one-point (default) | 3 (plate hourglass, w = +a,-a,+a,-a) | 44.69 % | 49.40 % |
| `mitc` assumed natural shear | 1 (in-plane only) | **2.35 %** | **0.75 %** |
| `full` equal-order (locking control) | 1 | 86.52 % | 70.96 % |

So the flat-facet family **does** pass this benchmark (`mitc`: 2.35 % < 3 % at
16x16, Richardson limit 2.23 %).  The operative defect is the rank deficiency of
the one-point shear block, and the library already carries the remedy
(`shell4_stiffness(..., plate_shear="mitc")`, exposed here as
`scordelis_lo_roof(..., plate_shear=...)`).  The default is deliberately left at
`sri` so every existing qualification keeps its numbers; flipping it suite-wide
is an open decision.  Guarded by
`tests/test_shell4_shear_rank_scordelis.py` and the rank census in
`tensorfem.shell4_rank`.

## Analysis

### Why Flat Facet Fails (superseded interpretation -- see update above)

The flat facet Shell4 element approximates the curved cylindrical surface using flat quadrilateral facets. This introduces several issues:

1. **Geometric approximation**: Adjacent facets have different normal vectors, creating artificial kinks in the geometry
2. **Membrane-bending decoupling**: The true curved-shell behavior requires membrane strains to couple with bending curvatures. Flat facets cannot capture this coupling properly
3. **Non-physical stress states**: The faceted geometry introduces spurious stress concentrations at facet edges

The non-monotone convergence (error actually increases from 8×8 to 32×32) indicates a fundamental limitation rather than a discretization issue.

### Why Cylindrical Shell4 Succeeds

The cylindrical shell4 element retains the exact cylindrical metric and curvature terms:

1. **Exact geometry**: The element uses (x, θ) parametric coordinates with exact mapping to (x, R sin θ, R cos θ)
2. **Proper curvature**: Curvature terms are retained in the strain-displacement relations
3. **Membrane-bending coupling**: The formulation properly captures the coupling between membrane action and bending in curved shells

The rapid convergence (0.02% error at 16×16) demonstrates that the element correctly represents the physics.

## Conclusions

1. **Flat facet elements are inadequate for curved shells**: The Shell4 element with flat facet approximation fails to capture the essential physics of curved-shell behavior, even with mesh refinement
2. **Specialized curved-shell elements are necessary**: The cylindrical shell4 element successfully captures membrane-bending coupling and achieves excellent accuracy
3. **Verification is essential**: This benchmark reveals a fundamental limitation that would not be apparent from flat-shell tests alone

## Recommendations

For curved shell problems:
- Use `cylindrical_shell4` for cylindrical geometries
- Develop specialized elements for other curved geometries (spherical, general curved shells)
- Do not rely on flat facet approximations for curved shells where membrane-bending coupling is important

## Reproduction

```bash
cd benchmarks/verified/fem_shell_scordelis_lo
OMP_NUM_THREADS=8 python3 run_verification.py
```

Outputs:
- `result.json`: Complete numerical results
- `gates_verified.json`: Gate verification status
- `scordelis_lo_convergence.png`: Convergence comparison plot

## References

1. Scordelis, A.C. & Lo, K.S. (1960). "Computer Analysis of Cylindrical Shells". Journal of the American Concrete Institute, 31, 539-560.
2. MacNeal, R.H. & Harder, R.L. (1985). "A proposed standard set of problems to test finite element accuracy". Finite Elements in Analysis and Design, 1(1), 3-20.
3. Belytschko, T., Liu, W.K., & Moran, B. (2000). "Nonlinear Finite Elements for Continua and Structures". Wiley.
