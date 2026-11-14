# CASE-FEM-ELASTIC-RESTRAINT-LEVY-VERIFICATION

**Anchor**: `GAP-PP-4-ICE-STRUCT-ELASTIC-RESTRAINT` — 板格边界的弹性嵌固转动刚度（纵骨/框架约束）
**Increment**: closes the *verification* debt left by
`docs/GAP-PP-4-ICE-STRUCT-ELASTIC-RESTRAINT.md` §1-9, which pinned only the two
ends of the band (`beta=0` against a Navier baseline, `beta=1` against
Timoshenko's clamped coefficients) and left the **interior** of the band with no
independent reference — the very range the manuscript JOUR-ICE-STRUCT §4.2 needs
to quote.

## 1. Case (what is being verified)

| item | value |
|---|---|
| panel | 1.0 m x 1.0 m x 10 mm steel plate (`t/a = 0.01`, so Mindlin shear error ~0.03%) |
| material | E = 200 GPa, nu = 0.3 |
| load | uniform pressure q = 1 kPa |
| restraint | `beta_x in {0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0}` on the x-edges, `beta_y = 0` (y-edges simply supported) |
| spring law | `K = zeta D / a`, `zeta = beta/(1-beta+eps)` (unchanged from the module) |
| manuscript panel (cross-check) | 0.8 m x 0.7 m x 22 mm, E = 206 GPa, 1 MPa patch over the full 0.7 m breadth, **all four** edges restrained |

Why `beta_y = 0`: it makes the boundary value problem *exactly* the one the
Levý series solves, so the comparison isolates the discretisation error of the
FE panel instead of mixing it with a reference that solves a different problem.

## 2. Method

Three references were built in `src/tensorfem/plate_levy_restraint.py`, none of
which uses finite elements:

1. `levy_restrained_plate` — **exact** Kirchhoff Levý solution. With
   `w = sum_n X_n(x) sin(n pi y/b)` the plate PDE reduces to
   `D (X'' - s_n^2 X)'' = q_n`; the homogeneous solution is written in a decaying
   exponential basis `e^{s(x-a)}, e^{-sx}, x e^{s(x-a)}, x e^{-sx}` and the
   4x4 boundary system (`X = 0`, `D(X'' - nu s^2 X) = ±K X'`) is solved in closed
   form per mode. `K = 0` is the simple support, `K -> inf` is clamped, so one
   routine gives an exact reference at **every** `beta`.
2. `kirchhoff_strip_end_moment` / `kirchhoff_strip_deflection` — beam
   closed forms for the cylindrical-bending limit, an independent check of (1).
3. `ritz_restrained_plate` — double-sine Rayleigh-Ritz with independent `Kx`,
   `Ky` (handles the four-edge manuscript panel, where (1) does not apply).

The FE side is the existing `solve_restrained_panel` (Mindlin Q4/SRI, SS2 hard
constraints, sparse direct solve) — untouched by this increment.

Toolchain: TensorFEM `agent/m20812`, numpy 2.x, scipy sparse solve; the sweep was
run on the cluster queue (job `cj2989`, node n330).

## 3. Results

### 3.1 Canonical plate: FEM vs exact Levý (centre values)

| beta | K [N·m/m/rad] | w_c exact [mm] | Mx_c exact [N·m/m] | w_c FEM 10x10 (err) | w_c FEM 20x20 (err) | w_c FEM 40x40 (err) | Mx_c FEM 40x40 (err) |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.000e+00 | 0.2218 | 47.886 | 0.2212 (-0.274%) | 0.2217 (-0.029%) | 0.2219 (+0.032%) | 47.870 (-0.033%) |
| 0.2 | 4.579e+03 | 0.2166 | 47.239 | 0.2159 (-0.315%) | 0.2165 (-0.038%) | 0.2167 (+0.030%) | 47.222 (-0.035%) |
| 0.4 | 1.221e+04 | 0.2089 | 46.278 | 0.2081 (-0.375%) | 0.2088 (-0.051%) | 0.2090 (+0.029%) | 46.261 (-0.037%) |
| 0.5 | 1.831e+04 | 0.2034 | 45.599 | 0.2026 (-0.418%) | 0.2033 (-0.060%) | 0.2035 (+0.028%) | 45.582 (-0.039%) |
| 0.6 | 2.747e+04 | 0.1963 | 44.705 | 0.1953 (-0.474%) | 0.1961 (-0.072%) | 0.1963 (+0.027%) | 44.686 (-0.041%) |
| 0.8 | 7.326e+04 | 0.1719 | 41.660 | 0.1707 (-0.663%) | 0.1717 (-0.110%) | 0.1719 (+0.027%) | 41.641 (-0.046%) |
| 1.0 | 1.832e+10 | 0.1047 | 33.245 | 0.1034 (-1.192%) | 0.1045 (-0.175%) | 0.1048 (+0.075%) | 33.230 (-0.045%) |

The exact reference spans `w_c = 0.2218 mm` (beta = 0) to `0.1047 mm`
(beta = 1), i.e. the band removes **52.8%** of the
simply-supported deflection — the quantity the manuscript has to bound.

### 3.2 Discretisation error vs exact reference (all meshes)

| beta | 10x10 (w / Mx) | 20x20 (w / Mx) | 40x40 (w / Mx) |
|---|---|---|---|
| 0.0 | -0.274% / -0.528% | -0.029% / -0.133% | +0.032% / -0.033% |
| 0.2 | -0.315% / -0.547% | -0.038% / -0.138% | +0.030% / -0.035% |
| 0.4 | -0.375% / -0.574% | -0.051% / -0.145% | +0.029% / -0.037% |
| 0.5 | -0.418% / -0.593% | -0.060% / -0.150% | +0.028% / -0.039% |
| 0.6 | -0.474% / -0.616% | -0.072% / -0.156% | +0.027% / -0.041% |
| 0.8 | -0.663% / -0.683% | -0.110% / -0.174% | +0.027% / -0.046% |
| 1.0 | -1.192% / -0.755% | -0.175% / -0.189% | +0.075% / -0.045% |

Max |error| on the finest mesh: **0.075%** (deflection),
**0.046%** (centre moment). Errors converge at the
expected `O(h^2)` rate (figure b) and change sign, so the residual is mesh, not bias.

### 3.3 Cylindrical-bending limit vs beam closed forms (b = 100 a)

| K [N·m/m/rad] | w Levý | w beam | err w | M_edge Levý | M_edge beam | err M |
|---|---|---|---|---|---|---|
| 0e+00 | 0.01302083 | 0.01302083 | +2.72e-07% | -0.000000 | -0.000000 | n/a (simple support) |
| 5e+00 | 0.00558036 | 0.00558036 | +1.48e-06% | -0.059524 | -0.059524 | +7.70e-08% |
| 5e+01 | 0.00300481 | 0.00300481 | +1.57e-06% | -0.080128 | -0.080128 | +3.73e-07% |
| 1e+06 | 0.00260419 | 0.00260419 | -3.61e-06% | -0.083333 | -0.083333 | +3.00e-06% |

Max |error| **3.61e-06%** (deflection), **3.00e-06%**
(moment): machine-level agreement with a completely independent formula, which
validates the Levý implementation itself (implementation bug would not care
about `b/a`).

### 3.4 Manuscript panel (all four edges) vs Rayleigh-Ritz

| beta | Kx [N·m/m/rad] | w_c Ritz [mm] | w_c FEM(80x70) [mm] | err w | Mx_c Ritz [N·m/m] | Mx_c FEM | err Mx | Ritz mode spread |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.000e+00 | 6.2252 | 6.2529 | +0.44% | 24361.5 | 24358.6 | -0.01% | +0.00% |
| 0.2 | 6.277e+04 | 5.9346 | 5.9625 | +0.47% | 23447.3 | 23472.6 | +0.11% | -0.00% |
| 0.4 | 1.674e+05 | 5.5281 | 5.5573 | +0.53% | 22168.5 | 22236.6 | +0.31% | -0.02% |
| 0.5 | 2.511e+05 | 5.2572 | 5.2880 | +0.58% | 21316.7 | 21415.5 | +0.46% | -0.04% |
| 0.6 | 3.766e+05 | 4.9188 | 4.9521 | +0.68% | 20252.7 | 20392.3 | +0.69% | -0.08% |
| 0.8 | 1.004e+06 | 3.9041 | 3.9500 | +1.18% | 17064.5 | 17343.8 | +1.64% | -0.30% |

`beta = 1.0` is **excluded** from this table: the Ritz sine basis is not
numerically resolvable at the fully closed limit (its own 30-vs-50-mode spread
diverges). `beta = 1` is instead covered exactly by §3.1 (Levý) and by
Timoshenko's CCCC coefficients.

## 4. Figures

* `benchmarks/artifacts/beta_levy_band.png` — (a) `w_c D/(qa^4)` vs `beta`:
  exact Levý curve with FEM points for 10/20/40 meshes; (b) log-log error vs
  mesh with a 2:1 reference slope.
* `benchmarks/artifacts/beta_levy_field.png` — (c) `M_x` along the sprung
  centreline at `beta = 0.5`, FEM vs exact; (d) the `M_x` field of the FEM panel
  at `beta = 0.5`, showing the moment migrating from mid-span to the frames.

## 5. Notes, limits, pitfalls

1. **Kirchhoff vs Mindlin model error.** The references are thin-plate; the FEM
   is Mindlin. At `t/a = 0.01` this is ~0.03%, and it is the reason the converged
   error is +0.03% (FEM softer) rather than 0.
2. **beta=1 is a regularised limit** (`zeta = beta/(1-beta+1e-6)`), not a hard
   clamp; both references were driven with the same finite `K`, so the comparison
   stays consistent.
3. **Mode truncation of the Levý series is an alternating series** in the centre
   quantities: the strip check needed `n = 6401` modes at `b/a = 100` to reach
   1e-6; using 41 modes there gives a spurious 1.5% "error".
4. **Ritz becomes ill-conditioned as the springs close** (the spring energy only
   constrains mode sums); the solve is row-scaled with a 1e-12 diagonal shift,
   and its truncation bias is quantified in the last column of §3.4 rather than
   hidden.
5. Scope: linear elastic, small deflection, uniform structured meshes, no
   foundation in the reference (the Winkler case is covered in the tributary
   case file).

## 6. Error judgement (is it < 3 %?)

| quantity | reference | worst error | < 3 %? |
|---|---|---|---|
| w_c, beta = 0 ... 1 (canonical) | exact Levý | 0.075% | **yes** |
| Mx_c, beta = 0 ... 1 (canonical) | exact Levý | 0.046% | **yes** |
| w_c, strip limit | beam closed form | 3.6e-06% | **yes** |
| M_edge, strip limit | beam closed form | 3.0e-06% | **yes** |
| w_c, manuscript panel, beta <= 0.8 | Ritz (variational) | 1.18% | **yes** |
| Mx_c, manuscript panel, beta <= 0.8 | Ritz (variational) | 1.64% | **yes** |

All key quantities are inside 3%; the canonical band (the one the paper quotes)
is inside **0.1%**.

## 7. Reproduction

```bash
python -m pytest tests/test_plate_levy_restraint.py -v         # 14 tests
python benchmarks/beta_levy_band_verification.py               # writes the JSON (~4 min)
python gen_levy_figures.py                                     # writes the two PNGs
```
