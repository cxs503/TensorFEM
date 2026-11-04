# Transient heat conduction: verification against closed-form solutions

Status: **verified** (all anchors pass). Evidence: `benchmarks/thermal_transient_reference_evidence.json`; generator: `benchmarks/thermal_transient_reference.py`.

## 1. Case and properties

| Quantity | Value |
|---|---|
| Length L | 1.0 m |
| Conductivity k | 1.0 W/(m K) |
| Density rho | 1.0 kg/m3 |
| Specific heat cp | 1.0 J/(kg K) |
| Diffusivity alpha = k/(rho cp) | 1.0 m2/s |
| Boundary conditions | T(0,t) = T(L,t) = 0 (Dirichlet step), or insulated |
| Initial condition | T = T0 = 1 (step) or sin(pi x/L) (single mode) |

## 2. Method

The merged public path `solve_transient_thermal` integrates `C Tdot + K T = f` with the theta-method. Two independent closed forms are used as references:

* **Step problem (Carslaw & Jaeger, Conduction of Heat in Solids, 2nd ed., Sec. 3.4)**: `T(x,t) = sum_{n odd} (4 T0/(n pi)) sin(n pi x/L) exp(-n^2 pi^2 alpha t / L^2)`, evaluated at the same nodes and time (400 odd modes retained).
* **Single sine mode**: `T = sin(pi x/L) exp(-pi^2 alpha t / L^2)` -- smooth data with no initial layer, so the spatial and temporal errors can be separated; the discrete initial vector equals discrete mode 1 exactly, hence no aliasing pollutes the dt study.

Element paths checked: LINE2 (1-D) and Q4 (2-D strip, one element through the thickness). All arithmetic is float64.

## 3. Results

### 3.1 Absolute accuracy at Fo = alpha t / L^2 = 0.10

| Mesh | steps | theta | relative L2 error | max abs error | gate | verdict |
|---|---|---|---|---|---|---|
| LINE2 32 | 1000 | 1.0 | 1.1085e-03 | 5.2418e-04 | 3% | PASS |

### 3.2 Spatial convergence (sine mode, Crank-Nicolson, 3000 steps)

| n elements | relative L2 error | observed order |
|---|---|---|
| 4 | 5.0390e-02 | 1.992 |
| 8 | 1.2667e-02 | 1.999 |
| 16 | 3.1699e-03 | 2.000 |
| 32 | 7.9267e-04 | -- |

### 3.3 Spatial convergence of the step problem (backward Euler, 2000 steps)

| n elements | relative L2 error | observed order |
|---|---|---|
| 8 | 2.5144e-02 | 2.036 |
| 16 | 6.1311e-03 | 2.181 |
| 32 | 1.3519e-03 | 3.120 |
| 64 | 1.5547e-04 | -- |

Richardson-extrapolated limit error at the finest mesh: **2.02e-05** (residual not attributable to the spatial discretisation).

### 3.4 Time-integration order (nx = 256, sine mode)

| scheme | steps | relative L2 errors | observed order | expected |
|---|---|---|---|---|
| theta=1.0 (backward Euler) | 4, 8, 16, 32 | 1.105e-01, 5.790e-02, 2.966e-02, 1.501e-02 | 0.982 | 1.0 |
| theta=0.5 (Crank-Nicolson) | 2, 4, 8 | 2.059e-02, 5.053e-03, 1.266e-03 | 1.997 | 2.0 |

The dt levels were chosen to stay above the fixed O(h^2) spatial floor (about 1e-6 at nx=256); finer CN levels were deliberately excluded because they would measure the spatial error, not the time integration.

### 3.5 Cross-checks

| Check | measured | gate | verdict |
|---|---|---|---|
| Q4 strip vs LINE2, same node set | 4.44e-14 | 1e-8 | PASS |
| Adiabatic mean temperature drift | 1.20e-14 | 1e-10 | PASS |
| sum(C) vs rho cp A L | 1.18e-16 | 1e-12 | PASS |
| max |K 1| (constant temperature null space) | 0.00e+00 | 1e-10 | PASS |

## 4. Error judgement against the acceptance standard

| Quantity | reference | error | < 3 % ? |
|---|---|---|---|
| T(x, Fo=0.10), LINE2 32, relative L2 | Fourier series | 0.111 % | yes |
| T(x, Fo=0.10), line 32, max nodal | Fourier series | 0.0005 (absolute, T0=1) | yes |
| peak temperature, adiabatic slab | conserved mean | 1.20e-14 | yes |

All key quantities are far below the 3% gate; the step-problem error at the coarsest useful mesh (16 elements) is 0.613 %, still inside the gate. The convergence rates reproduce the textbook values (2 in space, 1 for backward Euler, 2 for Crank-Nicolson).

## 5. Notes, limitations and pitfalls

* The pre-existing anchor `transient_rod_analytical_decay` (3 % gate) compares a single consistent-mass element with `tau = rho cp L^2/(3 k)`. That is a semi-discrete identity, not the PDE: it stays green even if the capacity matrix is scaled by a constant (both `K` and `C` scale together) and it cannot expose a wrong time-integration order. The anchors above close that gap; keep both, but treat the new ones as the governing-PDE evidence.
* The step-problem slope rises from 2.04 to 3.12 over the three refinements. A slope above 2 is the known 1-D consistent-mass superconvergence of the heat equation, not a super-convergent *claim* here: the text above quotes O(h^2) as the conservative asymptotic statement and the Richardson limit uses the measured slope, so 2.02e-05 is an upper bound on the non-spatial residual.
* **Setup trap 1**: an insulated slab initialised with `cos(pi x)` has zero mean temperature, so a *relative* drift metric divides by zero. The study uses `1 + 0.5 cos(pi x)` with an analytically known mean of 1.0.
* **Setup trap 2**: a Q4 strip clamped only at the two corner nodes leaves the other through-thickness nodes free, so the field becomes genuinely 2-D and disagrees with the 1-D answer by 34 %. Clamping all four face nodes makes the two discretisations agree to 4e-14.
* **Setup trap 3**: Crank-Nicolson with dt = O(h) on a fine mesh lets the O(h^2) spatial floor dominate and the measured slope collapses to about 0.9. The dt levels must sit above the spatial floor before any order is claimed.
* Scope: one-dimensional conduction with constant properties, Dirichlet and adiabatic boundaries, no radiation, no temperature-dependent k or cp, no convection boundary in the transient reference (convection is covered by the steady anchors in `docs/thermal-verification.md`).
* `theta < 0.5` is rejected by the solver (fail-closed); that behaviour is unchanged and still covered by `tests/test_thermal.py`.

## 6. Reproduction

```
PYTHONPATH=src python benchmarks/thermal_transient_reference.py
PYTHONPATH=src pytest tests/test_thermal_transient_reference.py -q
```

Figures: `docs/assets/thermal_transient_reference_profiles.png`, `docs/assets/thermal_transient_reference_convergence.png`.
