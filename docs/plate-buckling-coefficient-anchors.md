# Plate buckling coefficient: external anchors and an FEM k sweep

Mission 20817 - task `W29 plate-shell-buckling-closure item 2: external anchors + FEM k sweep` - repo `TensorFEM`
(generated 2026-11-14T19:26:26Z at commit `698cc45356e0`)

## 1. Case

A flat rectangular plate under uniaxial in-plane compression buckles at
`sigma_cr = k pi^2 D / (b^2 t)`. The coefficient `k` and the edge boundary
condition are the two ingredients every hull-plate check consumes, so the
case fixes one plate and sweeps the aspect ratio `alpha = a/b`:

| quantity | value |
| --- | --- |
| width `b` | 1.0 m |
| thickness `t` | 0.012 m |
| Young's modulus | 2.100e+11 Pa |
| Poisson ratio | 0.3 |
| yield stress (slenderness path only) | 3.550e+08 Pa |
| aspect ratios `a/b` | 0.5000, 0.6667, 1.0000, 1.3333, 1.5000, 2.0000, 2.5000, 3.0000, 4.0000 |
| mesh ladder `n_y` | 4, 6, 8, 12, 16, 20, 24, 32 |
| boundary conditions | simple support, clamped (two constraint realisations) |

## 2. Method

1. **Anchor acquisition.** The external anchors (DNV-RP-C201 plate buckling
   factors, Timoshenko & Gere plate coefficients, EN 1993-1-6 shell reduction
   curve) are re-attempted with recorded commands. Each is registered either
   `VERBATIM` (quote + locator + sha256 of the source text) or `WITHHELD` with
   **no** number; a remembered value is never recorded.
2. **Analytic closure.** The simply supported coefficient is computed twice by
   code paths that share no implementation - the closed-form minimisation
   `k = min_m (m/alpha + alpha/m)^2` and the repository's full-series Navier
   solver - and the residual is compared with `1e-10`.
3. **FEM sweep.** TensorFEM's Bogner-Fox-Schmit C1 Hermite plate element
   (`tensorfem.plate_buckling_hermite.bfs_plate_element_matrices` for bending,
   `tensorfem.plate_buckling_closure._bfs_element_gx` for the unit-compression
   geometric matrix) is assembled sparsely and the pencil `K v = mu G v` is
   solved for its smallest positive multiplier; `k = mu b^2/(pi^2 D)`. The
   sparse assembly is checked against the repository's dense path
   (max residual 3.382e-13).
4. **Boundary conditions.** Simple support fixes `w = 0` on all four edges;
   clamping is realised twice - elimination of the edge normal-slope dofs, and
   a near-rigid edge spring `k_r = 1e+08 D/b` - and both ladders are
   Richardson-extrapolated in `h`, so the clamping effect is a bracketed number
   rather than a single number from one mesh.
5. **Convergence order.** Measured from the FEM error at successive meshes
   against the exact analytic coefficient (simple support) or from
   three-grid differences (clamped, no exact reference).

## 3. Results

### 3.1 Anchors

- requested 3, verbatim 0, withheld 3 -> **HONEST_WITHHELD**
- acquisition attempts (command by command) are in the register under
  `anchors.acquisition.attempts`; no standard or handbook text exists in the
  execution environment, and the network probe returned no document.

### 3.2 Analytic closure

| alpha | k closed form | m | k Navier series | (m,n) | residual |
| --- | --- | --- | --- | --- | --- |
| 0.5000 | 6.2500000000 | 1 | 6.2500000000 | [1, 1] | 1.745e-16 |
| 0.6667 | 4.6944444444 | 1 | 4.6944444444 | [1, 1] | 4.646e-16 |
| 1.0000 | 4.0000000000 | 1 | 4.0000000000 | [1, 1] | 0.000e+00 |
| 1.3333 | 4.3402777778 | 1 | 4.3402777778 | [1, 1] | 3.768e-16 |
| 1.5000 | 4.3402777778 | 2 | 4.3402777778 | [2, 1] | 3.768e-16 |
| 2.0000 | 4.0000000000 | 2 | 4.0000000000 | [2, 1] | 0.000e+00 |
| 2.5000 | 4.1344444444 | 3 | 4.1344444444 | [3, 1] | 5.275e-16 |
| 3.0000 | 4.0000000000 | 3 | 4.0000000000 | [3, 1] | 0.000e+00 |
| 4.0000 | 4.0000000000 | 4 | 4.0000000000 | [4, 1] | 0.000e+00 |

max residual `5.275e-16` against tolerance `1e-10`; coefficient/stress round trip `2.148e-16`.

### 3.3 FEM k sweep and convergence

| alpha | k (exact, simple) | k FEM simple (finest) | order p | deviation | k clamped bracket | bracket width | k_clamped/k_simple |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5000 | 6.250000 | 6.250008272 | 3.98 | 1.324e-04% | [19.319142, 19.339887] | 0.1073% | 3.0927 |
| 0.6667 | 4.694444 | 4.694446065 | 3.79 | 3.452e-05% | [13.103531, 13.106606] | 0.0235% | 2.7916 |
| 1.0000 | 4.000000 | 4.000000258 | 4.00 | 6.446e-06% | [10.072258, 10.074021] | 0.0175% | 2.5183 |
| 1.3333 | 4.340278 | 4.340278029 | 4.01 | 5.799e-06% | [8.847323, 8.849550] | 0.0252% | 2.0387 |
| 1.5000 | 4.340278 | 4.340278574 | 3.99 | 1.836e-05% | [8.348914, 8.350549] | 0.0196% | 1.9238 |
| 2.0000 | 4.000000 | 4.000000258 | 4.00 | 6.446e-06% | [7.865527, 7.867104] | 0.0200% | 1.9666 |
| 2.5000 | 4.134444 | 4.134444919 | 3.99 | 1.148e-05% | [7.571750, 7.573102] | 0.0179% | 1.8315 |
| 3.0000 | 4.000000 | 4.000000258 | 4.00 | 6.447e-06% | [7.358015, 7.359366] | 0.0183% | 1.8397 |
| 4.0000 | 4.000000 | 4.000000258 | 4.00 | 6.446e-06% | [7.206531, 7.207890] | 0.0189% | 1.8018 |

### 3.4 Criteria

| id | requirement | verdict | measured |
| --- | --- | --- | --- |
| C1 | verbatim anchor locations >= 2, or an honest WITHHELD | HONEST_WITHHELD | verbatim 0, withheld 3 |
| C2 | analytic recompute residual <= 1e-10 | MEET | max residual 5.275e-16 |
| C3 | the influence of the edge boundary condition on k is quantified | MEET | widest clamped window 0.1073% |
| C4 | FEM mesh-convergence order reported and finest-mesh deviation from the analytic reference <= 3% (otherwise honest FAIL, not rounded away) | MEET | worst deviation 3.657e-02% (limit 3%), median order 4.00 |

### 3.5 Honest failures

None: every declared criterion is met. Nothing in this document is rounded
up, and no number was re-tuned after the results were seen.

## 4. Notes, limits, open items

- The anchors are **withheld**: no number here is a DNV-RP-C201 or
  Timoshenko & Gere value, and no comparison with those codes is claimed.
- The clamped coefficient has no closed form. It is a discretisation result
  from two constraint realisations of the **same** element, so the quoted
  window measures mesh/constraint consistency, not independent validation.
- `tensorfem.plate_buckling_closure.ritz_uniaxial_coefficient` builds its
  stiffness from `integral (W_,XX/alpha^2 + W_,YY)^2`, i.e. without the
  `-2(1-nu)(W_,XX W_,YY - W_,XY^2)` term. That term vanishes identically for
  simply supported sine modes (the module validates it at `k = 4.0`) but not
  for clamped modes, so it is an over-stiff context value for the clamped
  case, not an independent check.
- The independent Ritz route that was attempted (clamped-beam tensor basis,
  exact on the simply supported case at residual 6.84e-14) drifts from 10.063129 to 9.986665 between 12 and 24 modes and is therefore **not quoted**;
  its hypotheses and discriminating test are recorded in the sweep artifact
  (`exploratory_independent_route`, state OPEN).
- Scope: flat plate, constant thickness, Kirchhoff (thin) theory, uniaxial
  uniform compression, no prebuckling gradient, no shear, no plasticity, no
  post-buckling, no imperfection sensitivity.

## 5. Artifacts

- `benchmarks/anchors/plate_buckling_closure.json` - content sha256 `72be241ec375efa229f047f2c94dc28f854c1180d1e1682253c842206ffb0bfb`
- `artifacts/buckling_k_sweep.json` - content sha256 `ea2f48b4fe620f414e9446cc85bee0bfa13ea1ce8c0c2b7d3b8eb5ee5e2d6230`

