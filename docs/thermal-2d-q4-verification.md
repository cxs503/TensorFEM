# 2-D Q4 heat conduction: verification against closed-form solutions

Status: **PASS** -- evidence `benchmarks/thermal_2d_reference_evidence.json`, generator `benchmarks/thermal_2d_reference.py`, gate `tests/test_thermal_2d_reference.py`.

## 1. Why this report exists

`tensorfem.thermal` ships a two-dimensional element: bilinear `Q4`, 2x2 Gauss integration, consistent capacity, element-wise volumetric source and a two-node Robin (convection) edge. The merged path was verified only by

1. a linear-temperature patch test (`tests/test_thermal.py`), which any consistent element passes;
2. a 1-D series-resistance check, an analytical solution of a *different* problem;
3. `study_q4_strip_consistency`, which compares a Q4 strip against this code base's own 1-D solver -- a shared error in the source convention, the capacity distribution or the boundary measure cancels out.

Nothing in the suite compared the 2-D assembly with an external truth. This report does, on three independent references.

## 2. Case, geometry and parameters

### 2.1 Reference A -- plane wall, uniform generation, convective face

Geometry: `[0, L] x [0, H]` extruded wall of thickness `t` (out of plane). Boundary conditions: `T = 0` on `x = 0`; convection `-k dT/dn = h (T - T_inf)` on `x = L`; the two `y` faces are insulated (natural boundary condition, no data needed).

| Parameter | Value |
|---|---|
| Conductivity k | 45.0 W/(m K) |
| Volumetric source q | 5.0e+04 W/m3 |
| Convection coefficient h | 500.0 W/(m2 K) |
| Length L / height H / thickness t | 1.00 m / 0.50 m / 0.020 m |
| Ambient temperature T_inf | 0 K (temperature rise problem) |

Closed form (`-k T'' = q`, `T(0) = 0`, `-k T'(L) = h (T(L) - T_inf)`):

```
T(x) = -q x^2/(2k) + A x,   A = (q L + h q L^2/(2k)) / (k + h L)
```

With these numbers the wall slope is `A = 601.4271 K/m` and `T(L) = 45.8716 K`, with the temperature peak `161.8247 K` at `x = A k / q = 0.5413 m`. The surface value is **not** the textbook `q L / h = 100.0000 K`, which only holds in the `k -> infinity` limit: our first draft used the limit form, disagreed with the solver by 54% at `x = L` and by 82% in the L2 norm, and the solver was right (Sec. 5).

### 2.2 Reference B -- manufactured harmonic fields on a quad mesh

`T = x^2 - y^2` (quadratic) and `T = exp(pi x) sin(pi y)` (smooth), both harmonic; Dirichlet data taken from the exact field on all four edges. The quadratic field is nodally exact for bilinear elements on rectangles, so it is a machine-precision gate; the smooth field measures the h-convergence rate, on a structured and on a deliberately distorted mesh (`d = 0.06 L sin(pi x/L) sin(pi y/H)`, non-affine quads, positive Jacobians).

### 2.3 Reference C -- genuine 2-D transient cooling

Rectangle `1.0 x 0.5`, `k = rho = cp = 1` (`alpha = 1`), uniform initial field `T = 1`, zero temperature on all four edges. Separability gives the closed form as the product of two Carslaw & Jaeger 1-D step series:

```
T(x,y,t) = S(x,t;1.0) * S(y,t;0.5),
S(xi,t;L) = sum_{n odd} 4/(n pi) sin(n pi xi/L) exp(-n^2 pi^2 alpha t / L^2)
```

`t_end = 0.05`, i.e. `Fo = 0.05` in the long direction. Both directions are active, so the 2-D capacity and conductivity matrices are both exercised; the non-square domain removes the symmetry that an accidentally isotropic assembly could exploit.

## 3. Method and tool chain

* Solver: public path only -- `ThermalModel`, `solve_steady_thermal`, `solve_transient_thermal` (theta-method, theta = 0.5 here), `assemble_thermal` for the reaction check. Float64 throughout.
* Meshes: structured Q4 (`rect_mesh`), nodes ordered `i*(ny+1)+j`, counter-clockwise connectivity; distorted variant as in Sec. 2.2.
* References: closed forms evaluated at the same nodes and time; series truncated adaptively (400 odd modes) so the reference error is below 1e-14.
* Error norms: relative nodal L2 over all nodes (`||u_h - u|| / ||u||`) and the maximum absolute nodal error.
* Convergence: observed order `p = log(e_coarse/e_fine)/log(r)`. For the transient study the mesh and the time step are refined together (r = 2), so the measured rate is the coupled spatial/temporal rate; both terms are second order, hence the expected rate is 2.
* Independent energy audit for reference A: the Robin flux is re-evaluated by the trapezoidal rule on the nodal values with the *geometric* edge length and the prescribed thickness, and compared with the generated heat minus the heat leaving through the fixed face (from the reaction vector `K u - f`). Neither term is reused from the solver's own Robin matrix.

## 4. Results

### 4.1 Reference A: steady wall with source and convection

| Quantity | Value |
|---|---|
| relative L2 error (all nodes) | 9.294e-16 |
| maximum absolute error | 2.274e-13 K |
| T(L), Q4 | 45.8715596330 K |
| T(L), closed form | 45.8715596330 K |
| generated heat | 500.0000 W |
| heat out, convective face | 229.3578 W |
| heat out, fixed face | 270.6422 W |
| energy-balance relative error | 4.547e-16 |

### 4.2 Reference B: harmonic fields

| Study | property | value | gate |
|---|---|---|---|
| quadratic harmonic | max absolute nodal error | 2.220e-16 | < 1e-10 |
| smooth harmonic, structured | observed order | 1.936 | >= 1.7 |
| smooth harmonic, distorted | observed order | 1.944 | >= 1.7 |

| mesh (nx=ny) | structured, rel. L2 | order | distorted, rel. L2 | order |
|---|---|---|---|---|
| 4 | 2.3560e-02 | 1.813 | 2.3899e-02 | 1.826 |
| 8 | 6.7040e-03 | 1.885 | 6.7415e-03 | 1.898 |
| 16 | 1.8144e-03 | 1.936 | 1.8089e-03 | 1.944 |
| 32 | 4.7416e-04 | -- | 4.7029e-04 | -- |

### 4.3 Reference C: 2-D transient cooling

| nx x ny | steps | relative L2 error | observed order |
|---|---|---|---|
| 8 x 4 | 40 | 1.6224e-01 | 1.931 |
| 16 x 8 | 80 | 4.2557e-02 | 1.983 |
| 32 x 16 | 160 | 1.0765e-02 | -- |

Complementary structural checks on the same transient path:

| Check | measured | gate |
|---|---|---|
| pointwise bounds, minimum (maximum principle) | 0.000e+00 | >= -1e-12 |
| pointwise bounds, maximum | 0.137458 | <= 1 |
| centre-node relative error (backward Euler, coarse mesh) | 6.307e-03 | < 3e-02 |
| reflection symmetry error on a square domain | 1.665e-15 | <= 1e-12 |

Figures: `docs/assets/thermal_2d_wall_field.png` (field, exact field and wall profile), `thermal_2d_transient_field.png` (three times, numeric vs closed form), `thermal_2d_convergence.png` (h-refinement).

## 5. Pitfalls hit while writing this verification

* **The reference was wrong, not the solver.** The first closed form used the `k -> infinity` surface value `T(L) = q L/h`, which ignores the parabolic part of the profile and gives `100.0 K` instead of `45.9 K`. The relative error was 82% and the solver looked broken; re-deriving the convective condition with the parabolic term fixed it, and the solver then agreed to 1e-15. Always re-derive a reference before blaming the code.
* **Reaction sign.** For the heat equation a negative reaction at a fixed temperature means the constraint removes heat; the first version of the energy audit added instead of subtracted it and reported a 108% imbalance.
* **Polynomial harmonics are nodally exact.** `x^2 - y^2` (and `x^3 - 3xy^2`) come out at 1e-16 on any structured mesh, so they cannot measure a convergence order; they are used only as machine-precision gates. The order study needed a non-polynomial harmonic and, for a non-trivial case, a distorted mesh.
* **Thread oversubscription.** These systems have at most ~1200 unknowns, so the default torch thread pool (one worker per core) dominated the wall clock (>120 s versus 2 s with four threads). The benchmark caps the thread count.
* Scope: isotropic conductivity (the element accepts a scalar `k` only), linear (2-node) convection edges, `theta in [0.5, 1]`. Radiation, temperature-dependent properties and quadratic elements are outside this report.

## 6. Error verdict against the 3% accuracy standard

| Anchor | key quantity | error | gate | within gate? |
|---|---|---|---|---|
| A: wall with source + convection | relative L2 error | 9.294e-16 | 1e-8 | yes |
| A: wall with source + convection | energy balance | 4.547e-16 | 1e-8 | yes |
| B: quadratic harmonic | max absolute nodal error | 2.220e-16 | 1e-10 | yes |
| B: smooth harmonic (distorted) | relative L2 error, finest mesh | 4.703e-04 | 3% | yes |
| C: 2-D transient cooling | relative L2 error, finest mesh | 1.077e-02 | 3% | yes |
| C: 2-D transient cooling | relative L2 error, coarsest mesh | 1.622e-01 | 3% | **no** |
| C: maximum principle | pointwise bound violation | 0.000e+00 | 0 | yes |

Explicit answer to the 3% question:

* **Steady source + Robin face (A): yes**, far below it -- 9.3e-16 relative L2 and 4.5e-16 energy imbalance, i.e. machine precision for this problem.
* **Genuine 2-D transient (C) at the 32x16 mesh: yes**, 1.08%. At the coarsest 8x4 mesh it is 16.2% and does **not** meet 3%, which is the expected O(h^2) behaviour of a coarse mesh, not a solver defect: the observed order 1.98 over three refinements already forces the error to cross 3% as the mesh is refined.
* **Harmonic MMS (B): yes** -- 4.7e-04 on the finest distorted mesh, and order 1.94 / 1.94 (structured / distorted).
* No gate was relaxed to make an anchor pass; the only corrections made during this work were on the reference side (Sec. 5) and both are recorded there.

## 7. Reproduce

```
PYTHONPATH=src python benchmarks/thermal_2d_reference.py
PYTHONPATH=src python -m pytest tests/test_thermal_2d_reference.py -q
```
