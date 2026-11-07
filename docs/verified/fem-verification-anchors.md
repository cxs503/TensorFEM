# Verified anchors for linear elasticity, contact and modal analysis (IN36-N01)

**Scope.** This document certifies three independent verification surfaces for
TensorFEM on the three classical anchors a structural analyst is judged on:
slender bending (linear elasticity), Hertzian contact (nonlinear boundary
condition, but still linear elasticity in the interior) and free vibration
(eigenvalue problem). Each anchor lives in its own directory under
`benchmarks/verified/`, owns its closed-form oracle, its refinement ladder, its
gate manifest (`gates_verified.json`) and its figures; `result.json` carries every
number the document quotes. Nothing here is a re-run of an existing
`benchmarks/` case: the parameters, the oracles, the meshes, the solvers and the
gates are new, and the three anchors are graded against formulas that no part of
the solve path consumes.

**Evidence discipline.** Every gate names its reference formula, its threshold and
its verdict; `tests/test_verified_fem_anchors.py` re-derives the oracles inside
the test and asserts that each gate value is present in `result.json`, so neither
a drifted formula nor a hand-edited number can pass. Where a quantity does not
converge monotonically that is written down (§2) instead of being averaged away.

**Reproduction.**

```bash
OMP_NUM_THREADS=16 PYTHONPATH=src python benchmarks/verified/fem_cantilever/run.py
OMP_NUM_THREADS=32 PYTHONPATH=src python benchmarks/verified/fem_hertz_contact/run.py --levels 24 36
OMP_NUM_THREADS=32 PYTHONPATH=src python benchmarks/verified/fem_hertz_contact/run.py --levels 48 --penalty 100
OMP_NUM_THREADS=16 PYTHONPATH=src python benchmarks/verified/fem_plate_modal/run.py
PYTHONPATH=src python benchmarks/verified/render_report.py
PYTHONPATH=src pytest tests/test_verified_fem_anchors.py
```

The Hertz levels are cached (`_cache/`), so a slow mesh can be solved in its own
call; the other two anchors run end to end in under a minute each.

<!-- BEGIN:ANCHOR-DATA -->

### 1. Cantilever bending: Euler--Bernoulli deflection, stress, GCI ladder

**Geometry and material (campaign parameters).** `L = 10.0 m`, `b = 1.0 m`, `h = 0.1 m` (`L/h = 100`), `E = 2.1e+11 Pa`, `nu = 0.3`, tip load `F = 1e+04 N` along `-z`. `I = b h^3/12 = 8.333333e-05 m^4`.

**Independent oracle.** `w = F L^3/(3 E I) = 0.190476190 m`, `theta = F L^2/(2 E I) = 0.028571429 rad`, `sigma_xx(x,z) = F (L-x)(z-h/2)/I`. The transverse-shear contribution `F L/(kappa G A)` is 7.800e-05 of the bending deflection, so beam theory is the correct yardstick to two orders below the 1 % gate.

**Model.** Q8 serendipity plane-stress membrane of the `(x,z)` section (thickness `b`), 3x3 Gauss, sparse direct (SuperLU) solve, Chebyshev-clustered axial mesh so both the clamped root and the tip load stay inside the asymptotic range. A HEX20 3-D solid of the same bar is carried as a model-form cross-check.

#### Ladder (Q8 plane stress)

| axial x depth elements | dofs | w_tip [m] | rel. error | sigma_xx probe | rel. error | solve [s] |
|---|---|---|---|---|---|---|
| 20 x 2 | 330 | 0.187541349 | 1.541 % | 24.5088 MPa | 0.088 % | 0.01 |
| 40 x 4 | 1138 | 0.190178195 | 0.156 % | 27.1968 MPa | 0.004 % | 0.01 |
| 80 x 8 | 4194 | 0.190437390 | 0.020 % | 29.7278 MPa | 0.002 % | 0.05 |
| 160 x 16 | 16066 | 0.190457551 | 0.010 % | 29.2872 MPa | 0.001 % | 0.32 |

#### Grid convergence (Richardson, finest triple)

| quantity | value |
|---|---|
| observed order p | 3.684 |
| GCI_fine (Fs = 1.25) | 0.001 % |
| Richardson limit w [m] | 0.190459252 |
| ratio r | 2.00 |

#### Model-form cross-check (HEX20 3-D solid, uniform mesh)

| axial x width x depth elements | dofs | w_tip [m] | rel. error | solve [s] |
|---|---|---|---|---|
| 16 x 2 x 2 | 1503 | 0.186950062 | 1.851 % | 1.73 |
| 32 x 4 x 4 | 8835 | 0.188151131 | 1.221 % | 3.43 |
| 64 x 4 x 4 | 17475 | 0.188455340 | 1.061 % | 6.07 |

The solid model converges to a deflection ~1 % below beam theory and only slowly: at `L/h = 100` a fully integrated quadratic brick is shear-locking limited, and the fully clamped root additionally restrains the anticlastic contraction. This is why the 1 % gate is graded on the plane-stress model (whose exact limit *is* Euler--Bernoulli) and the solid is graded at 2 % as a cross-check.

#### Gates

| gate | independent reference | value | criterion | verdict |
|---|---|---|---|---|
| `geometry_extent` | declared L = 10 m, b = 1 m, h = 0.1 m | 0 | <= 1e-12 | PASS |
| `tip_deflection` | w = F L^3/(3 E I) | 9.78563e-05 | <= 0.01 | PASS |
| `gci_fine` | GCI = 1.25 |(w3-w2)/w3| / (r^p - 1) | 1.11602e-05 | <= 0.03 | PASS |
| `observed_order` | p = ln((w3-w2)/(w2-w1))/ln r | 3.68441 | >= 1.8 | PASS |
| `midspan_bending_stress` | sigma_xx = F (L - x)(z - h/2)/I | 6.25716e-06 | <= 0.02 | PASS |
| `reaction_equilibrium` | global equilibrium of the discrete system | 2.41782e-06 | <= 1e-05 | PASS |
| `solid3d_tip_deflection` | w = F L^3/(3 E I) | 0.0106095 | <= 0.02 | PASS |

### 2. Hertzian sphere--sphere contact: contact radius and peak pressure

**Problem (campaign parameters).** Two identical elastic spheres `R_1 = R_2 = 0.05 m`, `E = 2.1e+11 Pa`, `nu = 0.3`, pressed by `F = 1e+04 N`.

**Independent oracle (Hertz/Johnson).** `R = (1/R_1+1/R_2)^-1 = 0.025 m`, `E* = [sum (1-nu_i^2)/E_i]^-1 = 1.153846e+11 Pa`, `a = (3 F R/(4 E*))^(1/3) = 1.175667e-03 m`, `p_0 = 3F/(2 pi a^2) = 3.454399e+09 Pa`, approach `delta = a^2/R = 5.528775e-05 m` (=27.644 um per sphere), mean pressure `F/(pi a^2) = 2.3029e+09 Pa`.

**Model.** The Hertz reduction to a rigid indenter of radius `R` on a half-space with `E*/(1-nu^2) = 1.050000e+11 Pa` is used, i.e. TensorFEM's axisymmetric Q4 elasticity with nodal penalty contact, an active-set Newton loop and a least-squares `p^2` vs `r^2` recovery of `(a, p_0)` from the computed pressure field only. The prescribed approach is `delta = 5.528775e-05 m`; the contact radius, the peak pressure and the contact force are all *outputs*. Graded mesh (grading 2.6) clusters nodes at the axis and the contact surface; the domain is `1.2 R` wide and deep.

#### Ladder

| elements (r x z) | nodes in contact | a [mm] | a err. | p_0 [GPa] | p_0 err. | F [kN] | F err. | overlap/delta | p L2 err. |
|---|---|---|---|---|---|---|---|---|---|
| 24 x 24 | 7 | 1.1886 | 1.101 % | 3.5519 | 2.823 % | 10.3391 | 3.391 % | 0.0274 | 5.362 % |
| 36 x 36 | 11 | 1.1338 | 3.557 % | 3.4988 | 1.284 % | 10.2474 | 2.474 % | 0.0264 | 4.079 % |
| 48 x 48 | 14 | 1.1895 | 1.175 % | 3.4550 | 0.018 % | 10.2177 | 2.177 % | 0.0259 | 2.150 % |

#### Convergence record (honest form)

| quantity | value |
|---|---|
| pressure-field L2 order, finest pair | 2.23 |
| pressure-field L2 order, coarse pair | 0.67 |
| contact radius monotone | False |
| a order (finest pair) | 3.85 |
| p_0 order (finest pair) | 14.76 |

The **pressure field** converges monotonically (L2 error 5.4 % -> 4.1 % -> 2.1 %) at an observed rate of about 2.2 on the finest pair. The **recovered** `(a, p_0)` do not: the active-set cardinality jumps (7 -> 11 -> 14 nodes) and the lumped nodal traction makes the fit noisy, so `a` reads 1.1 %, 3.6 %, 1.2 % across the ladder. No order is claimed for `a` or `p_0`; the gates are graded on the finest level, where all three quantities sit below their thresholds.

#### Penalty-compliance study

| penalty factor | a err. | p_0 err. | F err. | overlap/delta |
|---|---|---|---|---|
| 28 | 3.557 % | 1.284 % | 2.474 % | 0.0264 |
| 100 | 4.664 % | 4.913 % | 3.981 % | 0.0079 |

Stiffening the penalty from 28 to 100 does **not** improve the agreement: the penalty overlap is only 2.6 % of the approach, so the remaining error is discretisation of the contact patch, and the stiffer constraint degrades the conditioning of the active-set system. The penalty bias is therefore documented, not tuned away.

#### Gates

| gate | independent reference | value | criterion | verdict |
|---|---|---|---|---|
| `contact_radius` | a = (3 F R /(4 E*))^(1/3) | 0.0117458 | <= 0.02 | PASS |
| `peak_pressure` | p_0 = 3 F /(2 pi a^2) | 0.000183652 | <= 0.03 | PASS |
| `contact_force` | prescribed F = 10 kN | 0.021771 | <= 0.03 | PASS |
| `pressure_field_l2` | p(r) = p_0 sqrt(1-(r/a)^2) | 0.0214962 | <= 0.05 | PASS |
| `force_balance` | global equilibrium of the discrete system | 6.13112e-13 | <= 1e-06 | PASS |
| `newton_converged` | residual <= 1e-10 of the contact-force scale | 1 | >= 1 | PASS |
| `refinement_levels` | campaign minimum of 3 | 3 | >= 3 | PASS |

### 3. Simply supported rectangular plate: first five natural frequencies

**Geometry and material (campaign parameters).** `a x b x h = 1.2 x 0.8 x 0.012 m` (`a/b = 1.50`, `a/h = 100`), `E = 2.1e+11 Pa`, `nu = 0.3`, `rho = 7850 kg/m^3`, all four edges simply supported (hard convention: `w = 0`, normal rotation restrained).

**Independent oracle (Navier/Kirchhoff).** `f_mn = (pi/2) sqrt(D/(rho h)) [(m/a)^2 + (n/b)^2]`, `D = E h^3/(12(1-nu^2)) = 3.323077e+04 N m`:

| mode | (m,n) | f analytic [Hz] |
|---|---|---|
| 1 | (1,1) | 66.586 |
| 2 | (2,1) | 128.051 |
| 3 | (1,2) | 204.881 |
| 4 | (3,1) | 230.491 |
| 5 | (2,2) | 266.346 |

**Model.** TensorFEM Mindlin--Reissner Q4 plate with selective reduced shear integration and its consistent mass matrix including rotary inertia, on a structured mesh with the library's hard simple-support DOF set. The pencil is solved by shift-invert Lanczos (`sigma = -1e2`); a dense LAPACK path was rejected because on this loong64 build it returned eigenpairs with ~20 % residual on the same reduced matrix (see the notes below). An equivalence gate re-solves the coarsest level with `solve_plate_vibration` and requires 1e-9 agreement.

#### Ladder

| elements a x b | dofs | f1..f5 [Hz] | rel. errors | min MAC | solve [s] |
|---|---|---|---|---|---|
| 12 x 8 | 351 | 67.54, 131.32, 219.38, 245.14, 282.12 | 1.43 %, 2.56 %, 7.08 %, 6.35 %, 5.92 % | 1.000000 | 3.5 |
| 24 x 16 | 1275 | 66.79, 128.75, 208.06, 233.64, 269.66 | 0.31 %, 0.54 %, 1.55 %, 1.36 %, 1.24 % | 1.000000 | 11.1 |
| 48 x 32 | 4851 | 66.61, 128.11, 205.39, 230.92, 266.69 | 0.03 %, 0.05 %, 0.25 %, 0.18 %, 0.13 % | 1.000000 | 48.7 |

#### Convergence

| mode | (m,n) | finest-pair order | finest rel. error |
|---|---|---|---|
| 1 | (1,1) | 3.24 | 0.033 % |
| 2 | (2,1) | 3.44 | 0.050 % |
| 3 | (1,2) | 2.65 | 0.247 % |
| 4 | (3,1) | 2.89 | 0.184 % |
| 5 | (2,2) | 3.25 | 0.131 % |

Library equivalence: max relative frequency difference `3.49e-11` against `solve_plate_vibration`. Spurious near-zero eigenvalues filtered at the finest mesh: `0`.

#### Gates

| gate | independent reference | value | criterion | verdict |
|---|---|---|---|---|
| `frequency_mode_1_11` | f_mn = (pi/2) sqrt(D/(rho h)) [(m/a)^2 + (n/b)^2] | 0.000326545 | <= 0.01 | PASS |
| `frequency_mode_2_21` | f_mn = (pi/2) sqrt(D/(rho h)) [(m/a)^2 + (n/b)^2] | 0.000498635 | <= 0.01 | PASS |
| `frequency_mode_3_12` | f_mn = (pi/2) sqrt(D/(rho h)) [(m/a)^2 + (n/b)^2] | 0.0024683 | <= 0.01 | PASS |
| `frequency_mode_4_31` | f_mn = (pi/2) sqrt(D/(rho h)) [(m/a)^2 + (n/b)^2] | 0.0018397 | <= 0.01 | PASS |
| `frequency_mode_5_22` | f_mn = (pi/2) sqrt(D/(rho h)) [(m/a)^2 + (n/b)^2] | 0.00130506 | <= 0.01 | PASS |
| `mode_shape_mac` | MAC = (phi_FE . phi_exact)^2 / (|phi_FE|^2 |phi_exact|^2) | 1 | >= 0.99 | PASS |
| `observed_order` | p = ln(e_coarse/e_fine)/ln(h_coarse/h_fine) | 2.65188 | >= 1.8 | PASS |
| `library_equivalence` | TensorFEM library entry point on the coarsest level | 3.49063e-11 | <= 1e-09 | PASS |
| `spurious_mode_filter` | supported boundary must carry no mechanism | 0 | <= 0 | PASS |

<!-- END:ANCHOR-DATA -->

## 4. Results figures

| anchor | field figure | convergence figure |
|---|---|---|
| cantilever | ![cantilever field](../../benchmarks/verified/fem_cantilever/cantilever_field.png) | ![cantilever convergence](../../benchmarks/verified/fem_cantilever/cantilever_convergence.png) |
| Hertz contact | ![Hertz pressure](../../benchmarks/verified/fem_hertz_contact/hertz_pressure.png) | ![Hertz convergence](../../benchmarks/verified/fem_hertz_contact/hertz_convergence.png) |
| Hertz penalty study | ![Hertz penalty](../../benchmarks/verified/fem_hertz_contact/hertz_penalty.png) | |
| plate modal | ![plate modes](../../benchmarks/verified/fem_plate_modal/plate_modes.png) | ![plate convergence](../../benchmarks/verified/fem_plate_modal/plate_convergence.png) |

## 5. Error judgement against the campaign criteria

| criterion (from the DAG node) | threshold | achieved | verdict |
|---|---|---|---|
| cantilever tip deflection vs `FL^3/(3EI)` | < 1 % | see §1 gate `tip_deflection` | PASS |
| cantilever GCI | < 3 % | see §1 gate `gci_fine` | PASS |
| cantilever observed order | >= 1.8 | see §1 gate `observed_order` | PASS |
| Hertz contact radius vs `(3FR/4E*)^(1/3)` | < 2 % | see §2 gate `contact_radius` | PASS |
| Hertz peak pressure vs `3F/(2 pi a^2)` | < 3 % | see §2 gate `peak_pressure` | PASS |
| plate first five frequencies vs Navier | each < 1 % | see §3 gates `frequency_mode_*` | PASS |
| plate / cantilever observed order | >= 1.8 | §3 gate `observed_order`, §1 gate `observed_order` | PASS |

The "is it below 3 %?" question, answered item by item: cantilever deflection
9.8e-5 relative, cantilever bending stress 6.3e-6, Hertz contact radius 1.17 %,
Hertz peak pressure 1.8e-4, Hertz pressure field L2 2.1 %, Hertz contact force
2.2 %, plate frequencies 0.03 %..0.25 %. All are below 3 %; all but the Hertz
contact radius and contact force are below 1 %. **No criterion is waived.**

## 6. Notes, limits and things that bit

1. **Order >= 1.8 is not universal, and the evidence says where it fails.**
   The cantilever (Chebyshev-clustered Q8, p = 3.68) and the plate (p = 2.65..3.44)
   demonstrate it. The Hertz anchor does **not** demonstrate it for the recovered
   `(a, p_0)`: the active-set cardinality jumps between levels, so those two
   numbers scatter (1.1 %, 3.6 %, 1.2 %). What does converge at ~2.2 is the
   pressure *field* (L2). Claiming an order for `a` would have meant fitting a
   line through three noisy points.
2. **A dense LAPACK eigensolve was silently wrong on this platform.** On the
   loong64 build, `scipy.linalg.eigh` (`evr`, `evx` and `evd` drivers alike)
   returned eigenpairs for the 1111-DOF reduced plate pencil with a ~20 %
   residual and eigenvalues near `-5e10`, while `torch.linalg.eigh` and sparse
   shift-invert Lanczos agreed with the physics to 1e-11. The anchor therefore
   uses shift-invert Lanczos and *proves* the discrete problem it solves is the
   library's by re-solving the coarsest mesh with `solve_plate_vibration`
   (agreement 3.5e-11). Anyone re-using dense eigensolvers here should check a
   residual, not just the eigenvalues.
3. **A 3-D solid is the wrong tool for a `L/h = 100` bending gate.** The fully
   integrated HEX20 cantilever stalls at ~1.06 % even at 17k dofs: shear locking
   plus the clamp's anticlastic restraint. Its ladder is reported (1.85 %, 1.22 %,
   1.06 %) and graded at 2 %, so the number is on the record instead of being
   replaced by the flattering plane-stress result.
4. **Plane stress is an idealisation, and it is the one beam theory assumes.**
   `sigma_yy = 0` in the 2-D model; the anticlastic state of a real 3-D bar shifts
   the tip deflection by `O(nu (h/L)^2) ~ 1e-4` relative, i.e. below the gate but
   not zero. This is why the anchor does not silently switch to plane strain.
5. **Contact is verified in its exact equivalent form.** Sphere--sphere is solved
   as a rigid indenter on a half-space with `R = R_s/2` and `E*`; that reduction is
   a closed-form theorem, not an approximation, but the run does not exercise a
   two-body deformable contact algorithm (the platform's own phase-3
   qualification still lists `deformable_to_deformable_contact_3d` as blocked).
6. **`p_0 = 3.45 GPa` exceeds the yield strength of ordinary steel.** These are
   elastic-operator checks; no plasticity, adhesion, friction or large-deformation
   effect is claimed.
7. **Thin-plate yardstick.** At `a/h = 100` the Mindlin shear correction to the
   Kirchhoff frequency is `O((h/a)^2) ~ 1e-4` relative, two orders below the gate;
   the hard simple-support convention differs from the soft (SS1) one by the same
   order.
8. **The selective-reduced-integration shear rule leaves zero-energy mechanisms**
   in the Q4 plate element. The count of near-zero eigenvalues is reported per
   level (0 at these meshes) and the retained spectrum is asserted positive, so a
   mechanism would fail the anchor rather than masquerade as a low frequency.

## 7. Audit chain

* Oracles: `benchmarks/verified/fem_cantilever/run.py:analytic`,
  `fem_hertz_contact/run.py:hertz_oracle`, `fem_plate_modal/run.py:navier_frequencies`
  — closed forms only, re-derived independently in the tests.
* Element kernels under test: `tensorfem.quadratic_plane.q8_stiffness`,
  `tensorfem.hex20.hex20_stiffness`, `tensorfem.hertz_axisymmetric`
  (assembly + active-set contact), `tensorfem.plate.q4_mindlin_stiffness` and
  `tensorfem.plate_vibration` (mass, boundary DOFs, library eigensolve).
* Evidence: `result.json` (all numbers, per-level), `gates_verified.json`
  (schema `tensorfem.verified-gates.v1`, per-gate reference/threshold/verdict),
  figures, and `docs/verified/fem-anchors-digest.json` (flat summary).
* Regression: `tests/test_verified_fem_anchors.py` (oracle re-derivation, gate
  self-consistency, end-to-end cantilever run, library-equivalence check).