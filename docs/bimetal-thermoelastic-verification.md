# Bimetal strip thermo-elasticity: Timoshenko (1925) closed form vs a 2-D Q9 FE model

* Node: **P189-N03** (TensorFEM, thermal-mechanical coupling)
* Generated: 2026-11-10T07:26:27Z (commit `1164e2d`, torch 2.7.1, python 3.11.6)
* Regenerate: `PYTHONPATH=src python benchmarks/bimetal_thermoelastic.py`
* Evidence: `benchmarks/bimetal_thermoelastic_evidence.json`

## 1. Case, physics and parameters

Two perfectly bonded layers of different thermal expansion form a strip that bends when heated while mechanically free. The temperature change produces a self-equilibrated field: an axial strain `eps0` plus a curvature `kappa`, with zero resultant force and zero resultant moment over the section. Timoshenko's 1925 analysis of bi-metal thermostats gives the curvature in closed form; the same expression is re-derived here from a 2x2 closed-section equilibrium (`section_equilibrium`), so a transcription slip in either form would be caught.

```
kappa = 6 (alpha1 - alpha2) dT (1 + m)^2 / ( h [ 3 (1+m)^2 + (1 + m n)(m^2 + 1/(m n)) ] )
m = t1/t2,  n = E1/E2,  h = t1 + t2,  eps_xx(y) = eps0 + kappa y,
w(x) = -kappa x^2/2  (built in at x = 0, free at x = L),  delta_tip = -kappa L^2/2
```

| quantity | value |
|---|---|
| h (total thickness) | 2.000 mm |
| L (strip length) | 40.00 mm (L/h = 20) |
| thickness ratio m = t1/t2 | 1 (t1 = 1.000 mm, t2 = 1.000 mm) |
| modulus ratio n = E1/E2 | 1 (E1 = 200.0 GPa, E2 = 200.0 GPa) |
| CTE | alpha1 = 1.70e-05 1/K (top), alpha2 = 1.20e-05 1/K |
| Poisson ratio | 0.3 (equal in both layers) |
| temperature rise dT | 100 K |
| closed-form curvature | 3.750000e-01 1/m |
| closed-form interface stress | -50.000 MPa (layer 1) / +50.000 MPa (layer 2) |
| closed-form free-end deflection | -300.000 um (-0.1500 h) |
| closed-form strain energy (unit width) | 1.250000e-01 J/m |

## 2. How it was done

* **Analytical side**: `analytical_solution` returns `(eps0, kappa)` from the 2x2 section equilibrium, cross-checks the Timoshenko closed form against it to 1e-10 relative and then builds the full field: strains, `sigma_xx(y)` in both layers, interface/surface extremes, and the strain energy `0.5 int E (eps - alpha dT)^2 dy` per unit width.
* **FE side**: `solve_bimetal_strip` assembles a structured **Q9** (biquadratic Lagrange, 3x3 Gauss, exactly integrated) plane-stress mesh with `ny_per_layer` elements per layer and a shared node line on the interface, applies the uniform `dT` as a thermal strain load `f = int B^T D eps_th dV`, clamps the whole `x = 0` edge, leaves every other edge traction free, and solves the sparse system (`scipy` SuperLU, float64). Q4 bilinear elements were *not* used: they lock in bending, so their deflection error measures spurious stiffness, not the discretisation of the exact solution.
* **Measured quantities**: free-end deflection `v(L, 0)`, mid-length deflection, `sigma_xx` at `(L/2, 0)` sampled separately in the layer below and above the interface, the through-thickness stress profile, the strain energy and the discrete thermo-elastic energy identity `U_el = C - u^T f_th/2`.
* **Mesh studies**: uniform and boundary-layer-graded (grading 1.15 toward the clamp) families with `n_x = 8 ... 128`; observed order and GCI from the grid triplets (Celik et al., *J. Fluids Eng.* 130 (2008) 078001); a `n_y` study to show the thickness direction carries no error; an `L/h = 10/20/40` study at fixed `dx/h` to quantify the clamped-edge model offset.

## 3. Results

### 3.1 Acceptance gates

| gate | criterion | required | measured | verdict |
|---|---|---|---|---|
| G0 | Timoshenko closed form vs independent 2x2 section equilibrium | rel. gap < 1e-10 | 1.04e-15 | PASS |
| G1 (a) | free-end deflection FEM vs closed form, m=n=1, dT=100 K | \|error\| < 3%, signed | -0.2803% | PASS |
| G2 (b) | max thermal stress (interface, both layers) vs closed form | \|error\| < 5% | 1.48e-11 | PASS |
| G3 (c) | 12 parameter cases (9 (m,n) + 3 dT): deflection and stress | all \|error\| < 5% | max deflection 0.773%, max stress 9.7e-09 | PASS |
| G4 (d) | observed order of the free-end deflection (4+ level GCI family) | p >= 1.8 | uniform p=2.02, graded p=3.40 | PASS |
| G5 (e) | single-material limit alpha1 = alpha2: curvature must vanish | \|FEM deflection\| / bimetal deflection < 1% | 9.10e-12 | PASS |
| G6 (f) | energy: strain energy vs closed form outside the clamp layer; thermo-elastic identity | region \|error\| < 2%; identity residual < 1e-6 | region -0.869%; identity 7.2e-09 | PASS |
| G8 (support) | section equilibrium of the FE stress field, sections >= 2h from the free end (the bimaterial corner singularity pollutes closer ones) | \|F\| / (E dalpha dT h) and \|M\| / (E dalpha dT h^2) < 1e-6 | F 2.0e-10, M 9.1e-09 | PASS |
| G7 (support) | through-thickness refinement (n_y = 2 ... 16) must not move the answer | \|spread\| < 0.05% | 2.55e-05 | PASS |

### 3.2 Reference case, accuracy mesh

Mesh: `n_x = 128`, `n_y = 8` per layer, 16962 dofs.

| quantity | closed form | FEM | relative error |
|---|---|---|---|
| free-end deflection [um] | -300.000000 | -300.840776 | -0.2803% |
| curvature implied by the tip [1/m] | 3.750000e-01 | 3.760510e-01 | +0.2803% |
| sigma_xx at (L/2, 0+) [MPa] | -50.000000 | -50.000000 | +1.478e-11 |
| sigma_xx at (L/2, 0-) [MPa] | +50.000000 | +50.000000 | +1.268e-11 |

### 3.3 Mesh convergence

Uniform family:

| n_x | dx_min/h | dofs | tip deflection [um] | error vs closed form | mid-span error |
|---|---|---|---|---|---|
| 8 | 2.5000 | 578 | -302.893284 | -0.9644% | -1.9987% |
| 16 | 1.2500 | 1122 | -301.631166 | -0.5437% | -1.1781% |
| 32 | 0.6250 | 2210 | -301.077709 | -0.3592% | -0.8019% |
| 64 | 0.3125 | 4386 | -300.891355 | -0.2971% | -0.6694% |
| 128 | 0.1562 | 8738 | -300.845493 | -0.2818% | -0.6332% |

Observed order from the grid triplets (uniform): 8-16-32: p = 1.189, 16-32-64: p = 1.570, 32-64-128: p = 2.023. Finest triplet GCI = 6.220e-05 of the tip deflection.

Graded family (1.15 towards the clamped edge):

| n_x | dx_min/h | tip deflection [um] | error vs closed form | mid-span error |
|---|---|---|---|---|
| 8 | 1.45700 | -301.805123 | -0.6017% | -1.3447% |
| 16 | 0.35895 | -300.864548 | -0.2882% | -0.6515% |
| 32 | 0.03466 | -300.774641 | -0.2582% | -0.6357% |
| 64 | 0.00039 | -300.774700 | -0.2582% | -0.6261% |
| 128 | 0.00000 | -300.774706 | -0.2582% | -0.6260% |

Observed order (graded): 8-16-32: p = 3.387, 16-32-64: p = n/a, 32-64-128: p = 3.405. Finest usable triplet GCI = 2.429e-09.

Interpretation: the Q9 element reproduces the through-thickness linear bending strain exactly, so refining `n_y` does not move the answer (see 3.4); the error is produced along `x`, where the clamped edge forces a Saint-Venant boundary layer of thickness `O(h)`. On a uniform family that layer is under-resolved at `n_x <= 64`, which is exactly the range where the observed order sits at 1.2-1.6; the asymptotic triplet (`n_x = 32, 64, 128`) reaches p = 2.02, and grading towards the clamp makes p = 3.40 with GCI 2.4e-09.

### 3.4 Through-thickness mesh independence

| n_y per layer | dofs | tip deflection [um] | error vs closed form | sigma error |
|---|---|---|---|---|
| 2 | 1746 | -300.947815 | -0.3159% | +1.6e-14 |
| 4 | 3298 | -300.941394 | -0.3138% | +3.2e-12 |
| 8 | 6402 | -300.940259 | -0.3134% | +4.9e-12 |
| 16 | 12610 | -300.940161 | -0.3134% | +9.7e-12 |

### 3.5 Slenderness: the price of the 2-D clamp

| L/h | n_x | dx/h | FEM tip [um] | closed form [um] | FEM/closed form | error |
|---|---|---|---|---|---|---|
| 10 | 25 | 0.400 | -75.532028 | -75.000000 | 1.007094 | -0.7094% |
| 20 | 50 | 0.400 | -300.931208 | -300.000000 | 1.003104 | -0.3104% |
| 40 | 100 | 0.400 | -1201.729516 | -1200.000000 | 1.001441 | -0.1441% |

The 2-D clamp (`u = v = 0` along the whole edge) is a *stronger* constraint than the beam's built-in end, so the FEM deflects slightly more than `-kappa L^2/2`; the offset halves when `L/h` doubles (it is `O(h/L)`, equivalently a small root rotation). It is a model difference, not a discretisation error, and it is reported instead of being tuned away.

### 3.6 Parameter sweep (9 `(m, n)` combinations + 3-`dT` ladder)

| m | n | dT [K] | kappa [1/m] | deflection FEM [um] | error | max |sigma| error |
|---|---|---|---|---|---|---|
| 0.5 | 0.5 | 100 | 2.797927e-01 | -224.641902 | -0.3608% | -9.33e-09 |
| 0.5 | 1 | 100 | 3.333333e-01 | -266.936098 | -0.1010% | -8.76e-09 |
| 0.5 | 3 | 100 | 3.732719e-01 | -297.564326 | +0.3527% | +4.13e-09 |
| 1 | 0.5 | 100 | 3.636364e-01 | -292.624088 | -0.5895% | +1.05e-09 |
| 1 | 1 | 100 | 3.750000e-01 | -300.840776 | -0.2803% | +1.27e-11 |
| 1 | 3 | 100 | 3.461538e-01 | -276.384921 | +0.1943% | +5.06e-10 |
| 2 | 0.5 | 100 | 3.648649e-01 | -294.149519 | -0.7734% | +9.66e-09 |
| 2 | 1 | 100 | 3.333333e-01 | -267.917056 | -0.4689% | -7.89e-09 |
| 2 | 3 | 100 | 2.403561e-01 | -192.499032 | -0.1114% | -5.10e-09 |
| 1 | 1 | 50 | 1.875000e-01 | -150.420388 | -0.2803% | +1.27e-11 |
| 1 | 1 | 100 | 3.750000e-01 | -300.840776 | -0.2803% | +1.27e-11 |
| 1 | 1 | 200 | 7.500000e-01 | -601.681552 | -0.2803% | +1.27e-11 |

### 3.7 Energy balance

| quantity | value |
|---|---|
| FEM strain energy, whole strip [J/m] | 2.426755e-01 |
| closed-form strain energy [J/m] | 1.250000e-01 |
| whole-strip relative error | +94.14% |
| thermal pool C = 0.5 int eps_th^T D eps_th dV [J/m] | 4.948571e+01 |
| 0.5 u^T f_th [J/m] | 4.924304e+01 |
| identity residual \|U_el - (C - u^T f_th/2)\| / U_el | 7.160e-09 |
| strain energy outside the clamp layer (x >= 2h) [J/m] | 1.115224e-01 |
| same region, closed form [J/m] | 1.125000e-01 |
| region relative error | -0.869% |
| fraction of the FEM energy outside the clamp layer | 0.460 |

The whole-strip energy is *not* usable as a gate: the pure-bending closed form is invalid inside the clamped boundary layer, where the strip is forced towards uniform axial strain and the energy density is an order of magnitude higher; that layer holds 54% of the total energy. Excluding `x < 2h` (the Saint-Venant length) the strain energy agrees to 0.87%. Note also that the mechanical Clapeyron relation `U = u^T f/2` does **not** hold for a thermal load: the exact identity is `U_el = C - u^T f_th/2` with `C` the thermal pool; the residual of that identity is 7.2e-09, i.e. solver round-off.

### 3.8 Single-material limit

| quantity | value |
|---|---|
| closed-form curvature at alpha1 = alpha2 | 0.000e+00 1/m |
| FEM tip deflection at alpha1 = alpha2 | -2.731e-15 m |
| FEM deflection / bimetal reference deflection | 9.104e-12 |
| FEM residual interface stress vs 50 MPa scale | 8.61e-13 |

### 3.9 Section equilibrium of the FE stress field

The free end carries no traction, so the internal force and moment resultants of the exact solution vanish on every section outside the clamped layer. Integrating the *FE* stress field (two interior samples per layer pin the linear stress, then the polynomial integrals are evaluated analytically):

| section | axial force / (E dalpha dT h) | moment / (E dalpha dT h^2) |
|---|---|---|
| x = L/2 | 6.36e-12 | 5.40e-10 |
| x = 0.9 L | 2.00e-10 | 9.13e-09 |
| x = 0.97 L | 9.19e-06 | 1.94e-04 |

Trap met here: a trapezoidal rule on `y * sigma_xx` is *not* exact (the integrand is quadratic in `y`) and leaves a mesh-independent spurious moment of 1.6e-04 of the scale *at every mesh*, which looks exactly like a real equilibrium violation. The integrals are therefore done analytically.

The `x = 0.97 L` row is kept on purpose: the bimaterial corner at the free end carries a stress singularity, so the *recovered* stress within about one thickness of it is not in balance and this check is only used from `x = 0.9 L` towards the clamp. Pointwise stresses in that corner must not be compared with the closed form at all.

## 4. Figures

![Field cloud of sigma_xx on the strip (Q9 plane stress) with the bending sign convention and the mid-line deflection.](bimetal_sigma_field.png)

*Field cloud of sigma_xx on the strip (Q9 plane stress) with the bending sign convention and the mid-line deflection.*

![Through-thickness sigma_xx at mid-length: the piecewise-linear closed form and the FEM sample, including the interface jump.](bimetal_section_stress.png)

*Through-thickness sigma_xx at mid-length: the piecewise-linear closed form and the FEM sample, including the interface jump.*

![h-refinement of the free-end deflection (uniform and graded families) with the observed order and a slope -2 guide.](bimetal_convergence.png)

*h-refinement of the free-end deflection (uniform and graded families) with the observed order and a slope -2 guide.*

![Mid-line deflection along the strip and the residual against the closed form (a small root rotation from the 2-D clamp).](bimetal_deflection_profile.png)

*Mid-line deflection along the strip and the residual against the closed form (a small root rotation from the 2-D clamp).*

![Relative errors of the 12 parameter cases against the acceptance gates (log scale).](bimetal_sweep_errors.png)

*Relative errors of the 12 parameter cases against the acceptance gates (log scale).*

## 5. Notes, limitations and traps

1. **Sign convention.** With layer 1 (larger CTE) on top, `kappa > 0` means the top layer ends up on the outside of the arc, i.e. `w'' = -kappa` and the free end dips. The FEM reproduces that sign; a sign flip would show up as a 200 % error.
2. **The clamp is not the beam's built-in end.** `u = v = 0` on the whole edge adds a root rotation, so the FEM tip deflection is ~0.26 % above the closed form at `L/h = 20` and the offset scales as `O(h/L)`. Chasing it with mesh refinement does not work (it is not a discretisation error) - refining the strip length does.
3. **The interface stress is discontinuous, and singular at the free ends.** The closed form jumps at `y = 0`; pointwise stresses near `x = 0` and `x = L` are not comparable with it, which is why the stress gate is sampled at mid-length only.
4. **Q4 would have made the convergence order meaningless.** Bilinear plane-stress elements lock in bending; the deflection would converge from a spurious stiff limit.
5. **Thermal loads break the mechanical Clapeyron relation.** `U_el = u^T f/2` is wrong by a factor ~200 here; the correct discrete identity carries the thermal pool.
6. **Scope.** Linear elasticity, temperature-independent `E, alpha`, equal `nu` in both layers, small strain (max rotation 0.015 rad), no plasticity, no creep, no buckling of the compressed layer, no transient conduction (the temperature field is uniform by assumption).

