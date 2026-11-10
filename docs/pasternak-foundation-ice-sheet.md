# Ice sheet on a Pasternak foundation: bending response and the local → global failure transition

## Document information
- **Mission / node**: p485w1-n02-fem-pasternak
- **Repository**: TensorFEM (finite-element methods and solver-specific verification)
- **Modules**: `tensorfem.pasternak_beam`, `tensorfem.ice_sheet_pasternak`
- **Evidence file**: `artifacts/pasternak_ice_sheet/pasternak_ice_sheet_benchmark.json`
- **Driver**: `benchmarks/pasternak_ice_sheet_bending.py`
- **Figures**: `docs/figures/pasternak_*.png`
- **Reproduce**: `python benchmarks/pasternak_ice_sheet_bending.py`, then
  `python -m pytest tests/test_pasternak_beam.py tests/test_ice_sheet_pasternak.py -q`
- **Status**: all three quantitative gates met (§5)

## 1. Physical background and parameters

A level ice sheet is classically a plate on a Winkler bed of buoyancy
`k = rho_w g`. That bed carries no shear: neighbouring springs are independent.
Two situations in which shear transfer in the foundation is real are ice over a
rubble / granular subgrade (ice roads, landfast ice over shoals, ice pads) and
ice inside a rubble field. The Pasternak (two-parameter) bed adds a shear layer of
modulus `G_s` [N/m]:

    p(x) = k w(x) - G_s w''(x)      (strip form, per unit width)

so the sheet obeys

    D w'''' - G_s w'' + k w = q,    D = E h^3 / (12 (1 - nu^2)).

`G_s = 0` recovers the Winkler / Hetenyi problem. The characteristic equation
`D l^4 - G_s l^2 + k = 0` has two decaying roots. For `G_s > 2 sqrt(D k)` they are
real and the response is a sum of two monotone layers: a fast *shear* layer
`l_shear -> sqrt(D/G_s)` and a slow *soft* layer `l_soft -> sqrt(G_s/k)`. Below
that threshold the roots are a complex pair and the decay is oscillatory, i.e.
the Hetenyi cosine-sine pattern.

| parameter | value | unit |
|---|---|---|
| ice thickness h | 1.00 | m |
| Young's modulus E | 5.000e+09 | Pa |
| Poisson ratio nu | 0.30 | - |
| ice / water density | 917.0 / 1025.0 | kg/m^3 |
| gravity g | 9.81 | m/s^2 |
| flexural strength sigma_f | 500000.000 | Pa |
| crushing strength sigma_c | 2.000e+06 | Pa |
| plate rigidity D | 4.578755e+08 | N m |
| buoyancy stiffness k = rho_w g | 10055.250 | N/m^3 |
| freeboard (1 - rho_i/rho_w) h | 0.1054 | m |
| first-crack moment sigma_f h^2 / 6 | 83333.3 | N m/m |
| classical ice length (D/k)^(1/4) | 14.61 | m |
| Hetenyi length (4D/k)^(1/4) | 20.66 | m |

The elastic and geometric values are textbook sea-ice values; sigma_f and
sigma_c are **inputs**, not calibrated against a test series (§6).

## 2. Method and tool chain

1. **Weak form.** `int EI w'' v'' dx + int k w v dx + int G_s w' v' dx = int q v dx`.
   The third term is the new one: `G_s int N'^T N' dx` on the C1 Hermite cubic
   element (4x4, rank 3 -- a constant deflection carries no shear energy). The
   first two reuse the existing Winkler machinery
   (`tensorfem.beam_foundation.eb_beam_stiffness`,
   `hermite_foundation_stiffness`); the plate-level Pasternak model in
   `tensorfem.foundation_plate_bending` was left untouched.
2. **Assembly and solve.** Two-node elements make the global matrix exactly block
   tridiagonal, so it is stored as 2x2 nodal blocks and solved by a block-Thomas
   sweep (linear cost; 9·10^4-element meshes are affordable). The sweep is
   cross-checked against a dense `torch.linalg.solve` in the tests. Prescribed
   zero dofs are eliminated by zeroing the row/column plus a unit pivot.
3. **Closed form for the infinite beam.** With `l_f, l_s` the decaying roots and
   `A = P / (2 EI l_f (l_s^2 - l_f^2))`, `B = -P / (2 EI l_s (l_s^2 - l_f^2))`:
   `w(x) = A exp(-l_f |x|) + B exp(-l_s |x|)`. This reduces algebraically to
   Hetenyi's `(P beta / 2k) exp(-beta|x|) (cos beta x + sin beta|x|)` at
   `G_s = 0` (verified to round-off in the tests) and gives
   `w(0) = P / (2 EI l_f l_s (l_f + l_s)) -> P / (2 sqrt(k G_s))` and
   `M(0) = -EI w''(0) = P / (4 l)` for large `G_s`.
4. **Mode extraction from the FE field.** Uniformly spaced samples of a
   two-exponential field satisfy `w_n = p w_(n-1) - q w_(n-2)` with
   `p = exp(-l_f h) + exp(-l_s h)`, `q = exp(-(l_f+l_s) h)`. A least-squares fit
   of `(p, q)` (Prony / linear prediction) and the roots of `z^2 - p z + q`
   recover **both** decay rates from the FE solution alone; the fast rate is read
   from a near-field window of four shear lengths.
5. **Failure model.** Three idealised mechanisms per metre of sheet width:
   contact crushing `sigma_c 2a`, first-crack bending `M_cap P_ref / M_max` from
   the strip FE solution, and foundering `rho_w g (1 - rho_i/rho_w) h L`.
6. **Tool chain.** torch 2.7, float64 only -- no new dependency. Coverage is
   measured with `python -m coverage run -m pytest` because the pytest-cov plugin
   is incompatible with this torch build (pre-existing environment issue).

## 3. Results

### 3.1 (a) Peak deflection against the analytic anchor

Infinite strip, point load P = 1·10^5 N/m, free ends, domain
`12 max(l_shear, l_soft) + 12 l_winkler`, mesh ~ l_shear/12.

| G_s [N/m] | regime | l_shear [m] | l_soft [m] | elements | w_FEM(0) [m] | anchor [m] | error | moment error | reaction/load |
|---|---|---|---|---|---|---|---|---|---|
| 0.000e+00 | oscillatory | 20.659 | 20.66 | 400 | 0.240699 | 0.240699 | -5.37e-06 % | -1.372 % | 1.000000 |
| 1.000e+07 | monotone | 6.937 | 30.76 | 1068 | 0.131897 | 0.131897 | -5.28e-07 % | -1.173 % | 1.000000 |
| 5.000e+07 | monotone | 3.029 | 70.45 | 4332 | 0.067672 | 0.067672 | +3.72e-05 % | -0.998 % | 1.000000 |
| 2.000e+08 | monotone | 1.513 | 141.02 | 15387 | 0.034886 | 0.034886 | +1.88e-04 % | -0.967 % | 1.000000 |
| 1.000e+09 | monotone | 0.677 | 315.36 | 40000 | 0.015734 | 0.015734 | +5.41e-04 % | -1.737 % | 1.000000 |

Worst peak-deflection error **+5.41e-04 %** (acceptance < 1 %). The peak
deflection is node-superconvergent while the bending moment converges more slowly
(second-derivative quantity):

| elements per l_shear | elements | dofs | deflection error | moment error |
|---|---|---|---|---|
| 2 | 2565 | 5132 | +1.68e-04 % | -6.197 % |
| 4 | 5129 | 10260 | +2.11e-04 % | -2.994 % |
| 6 | 7694 | 15390 | +2.12e-04 % | -1.967 % |
| 10 | 12823 | 25648 | +2.17e-04 % | -1.165 % |
| 16 | 20516 | 41034 | +1.91e-04 % | -0.722 % |

At `G_s = 0` the closed form reproduces the pre-existing
`hetenyi_point_load_deflection` value to round-off (relative difference
2.306e-16), so the new implementation and the established anchor agree
bit-for-bit.

### 3.2 (b) Characteristic-length scaling

Both lengths are extracted from the FE field (Prony fit), not from the analytic
formula:

| G_s [N/m] | G_s / 2 sqrt(D k) | elements | l_shear FE [m] | analytic [m] | error | l_soft error |
|---|---|---|---|---|---|---|
| 1.000e+08 | 23.3 | 5710 | 2.13844 | 2.14030 | -0.087 % | +1.66 % |
| 2.000e+08 | 46.6 | 11304 | 1.51249 | 1.51316 | -0.044 % | +1.64 % |
| 4.000e+08 | 93.2 | 22490 | 1.06968 | 1.06992 | -0.022 % | +1.63 % |
| 8.000e+08 | 186.4 | 44861 | 0.75645 | 0.75654 | -0.011 % | +1.62 % |
| 1.600e+09 | 372.8 | 89601 | 0.53492 | 0.53495 | -0.005 % | +1.64 % |

Power-law fits:
`l_shear = 21304.3287 G_s^(-0.499792)` (r^2 = 0.99999999) -> exponent
error **2.07568e-04** (acceptance < 0.02), and
`l_soft = 0.010141 G_s^(0.499970)` (r^2 = 0.99999998).

**Physics note (honest deviation from the node wording).** The node asks for
`l_c ~ G_s^(-1/2)`. That law holds for the *shear boundary layer*
`l_shear = sqrt(D/G_s)`. The *dominant* spreading length in the strong-shear
regime is the complementary root `l_soft = sqrt(G_s/k)`, which grows as
`G_s^(+1/2)`; both are reported and both fits are clean. The Hetenyi length
`(4D/k)^(1/4)` is independent of `G_s`.

### 3.3 (c) Failure-mode transition versus l_c/L

Sheet-length sweeps at five bed-shear values (`G_s = 0 ... 100 · 2 sqrt(D k)`),
contact half-length a = 0.5 m, `l_c` = dominant decay length:

| G_s [N/m] | l_c [m] | flexure plateau [N/m] | plateau spread | infinite-beam analytic | FEM - analytic | transition l_c/L |
|---|---|---|---|---|---|---|
| 0.000e+00 | 20.66 | 16299.6 | +1.438 % | 16135.2 | +1.019 % | 0.413 |
| 4.291e+06 | 14.61 | 23470.0 | +2.096 % | 22818.7 | +2.854 % | 0.292 |
| 2.146e+07 | 45.96 | 41485.6 | +1.383 % | 39523.1 | +4.966 % | 0.919 |
| 8.583e+07 | 92.36 | 80891.8 | +2.063 % | 73940.9 | +9.401 % | 0.924 |
| 4.291e+08 | 206.58 | 199283.4 | n/a | 162157.0 | +22.895 % | 1.033 |

The flexure plateau is independent of sheet length (spread
+2.096 % over the `l_c/L <= 0.25` window): the classical result that a
long sheet's capacity does not depend on its size. Foundering-branch exponents,
fitted per family (the branch is linear in L, i.e. -1 in `l_c/L`):

| G_s [N/m] | points | fitted exponent | r^2 |
|---|---|---|---|
| 0.000e+00 | 2 | -1.000000 | 1.00000000 |
| 4.291e+06 | 2 | -1.000000 | 1.00000000 |
| 2.146e+07 | 2 | -1.000000 | 1.00000000 |
| 8.583e+07 | 3 | -1.000000 | 1.00000000 |
| 4.291e+08 | 4 | -1.000000 | 1.00000000 |

Contact-mechanism sweep at L = 400 m, `G_s = 0` (per metre of width):

| a [m] | a/l_c | local crushing [N/m] | global flexure [N/m] | foundering [N/m] | mode |
|---|---|---|---|---|---|
| 5.0000e-04 | 2.42028e-05 | 2000.0 | 16222.4 | 423792.0 | local_crushing |
| 0.0010 | 4.84057e-05 | 4000.0 | 16222.4 | 423792.0 | local_crushing |
| 0.0020 | 9.68114e-05 | 8000.0 | 16222.4 | 423792.0 | local_crushing |
| 0.0040 | 1.93623e-04 | 16000.0 | 16222.4 | 423792.0 | local_crushing |
| 0.0080 | 3.87245e-04 | 32000.0 | 16222.4 | 423792.0 | global_flexure |
| 0.0200 | 9.68114e-04 | 80000.0 | 16222.4 | 423792.0 | global_flexure |
| 0.0500 | 0.00242 | 200000.0 | 16222.4 | 423792.0 | global_flexure |
| 0.1000 | 0.00484 | 400000.0 | 16222.4 | 423792.0 | global_flexure |
| 0.5000 | 0.02420 | 2.0e+06 | 16533.9 | 423792.0 | global_flexure |
| 1.0000 | 0.04841 | 4.0e+06 | 16944.7 | 423792.0 | global_flexure |
| 2.0000 | 0.09681 | 8.0e+06 | 17808.6 | 423792.0 | global_flexure |

The local <-> global crossing is bracketed at
a in [0.004, 0.008] m: **contact crushing governs only for contact
half-lengths of a few millimetres** with the declared strengths
(sigma_c/sigma_f = 4, h = 1 m). Over the practical contact range, floating-ice
capacity is flexure-governed, which is the classical result. The length-scale
transition that does occur is flexure -> foundering, and its position moves with
`G_s` (`l_c/L` about 0.3 ... 1.0).

Patch-size convergence of the plateau towards the point-load analytic value
(`G_s = 5 · 2 sqrt(D k)`, L = 500 m, mesh tied to the patch, h = 2a/6). The
FEM - analytic gap in the table above is therefore the finite patch, not a model
error, and it closes linearly with `a/l_shear`:

| a [m] | a/l_shear | elements | FEM capacity [N/m] | point-load analytic | difference |
|---|---|---|---|---|---|
| 0.500 | 0.108 | 3000 | 41950.5 | 39523.1 | +6.142 % |
| 0.200 | 0.043 | 7500 | 40481.0 | 39523.1 | +2.424 % |
| 0.100 | 0.022 | 15000 | 40000.0 | 39523.1 | +1.207 % |
| 0.050 | 0.011 | 30000 | 39760.1 | 39523.1 | +0.600 % |
| 0.020 | 0.004 | 30000 | 39609.2 | 39523.1 | +0.218 % |

### 3.4 Corroboration with the pre-existing plate-level Pasternak solution

`tensorfem.foundation_plate_bending.infinite_plate_two_parameter_point_load`
(unchanged, pre-existing) evaluated with the ice properties and P = 1·10^5 N:

| G_s [N/m] | plate peak deflection [m] | relative to Winkler |
|---|---|---|
| 0.000e+00 | 0.005826 | 1.0000 |
| 1.000e+07 | 0.002625 | 0.4505 |
| 1.000e+08 | 6.119162e-04 | 0.1050 |
| 1.000e+09 | 9.779019e-05 | 0.0168 |
| 1.000e+10 | 1.344362e-05 | 0.0023 |

The strip is the 1-D limit of that model: bed shear spreads the load and the peak
deflection drops monotonically. No new 2-D solver was written.

### 3.5 Strip field (h = 1 m, L = 240 m, a = 1 m, P = 1·10^5 N/m, G_s = 0)

Foundation reaction split: Winkler 99999.999 N/m, end traction
0.000e+00 N/m, applied load 100000.0 N/m (equilibrium error
-0.000 %). The moment and buoyancy-reaction diagrams are in
`docs/figures/pasternak_ice_sheet_fields.png`.

## 4. Figures

| file | content |
|---|---|
| `docs/figures/pasternak_peak_error.png` | (a) FE vs closed-form deflection profiles and the peak-error gate |
| `docs/figures/pasternak_length_scaling.png` | (b) FE-extracted l_shear, l_soft vs G_s with the G_s^(-+1/2) references |
| `docs/figures/pasternak_failure_mode_map.png` | (c) critical load vs l_c/L per bed shear, plus the mechanism ranking vs contact size |
| `docs/figures/pasternak_ice_sheet_fields.png` | 2-D plate deflection field (Winkler vs Pasternak) and the strip moment / buoyancy diagrams |

## 5. Acceptance gates

| criterion | result | measured | threshold |
|---|---|---|---|
| (a) FE peak deflection vs analytic anchor | PASS | +5.41e-04 % | < 1 % |
| (b) l_shear ~ G_s^m, |m + 1/2| | PASS | 2.07568e-04 | < 0.02 |
| (c) flexure plateau spread / foundering slope | PASS | +2.096 % / -1.000000 | < 5 % / -1 +- 0.05 |

Criterion (d), the pytest + coverage gate:

- `python -m pytest tests/test_pasternak_beam.py tests/test_ice_sheet_pasternak.py -q`
  -> **33 + 21 = 54 passed**;
- `python -m coverage run --source=src/tensorfem -m pytest <both files> -q` then
  `python -m coverage report` -> `pasternak_beam.py` **95 %**,
  `ice_sheet_pasternak.py` **98 %**, combined **96 %** (gate >= 85 %).

## 6. Caveats, limitations and honest failures

1. **Strength calibration.** sigma_f = 0.5 MPa and sigma_c = 2 MPa are textbook
   values. The mechanism *ranking* follows from mechanics, but no sea-ice test
   series was used to calibrate absolute failure loads, and the strong
   strain-rate dependence of ice strength is outside this model. The failure
   loads are model values, not field predictions.
2. **1-D strip idealisation.** The failure analysis is a unit-width strip; the
   2-D plate Pasternak closed form is used only as an implementation cross-check
   and no 2-D failure surface was computed.
3. **Literature comparison is asymptotic, not point-wise.** The classical limits
   used are Hetenyi (1946) for `G_s = 0`; Pasternak (1954) / Vlasov & Leont'ev
   (1960) for the two-parameter bed; Kerr (1964, 1976) and Michel (1978) for the
   local-crushing vs global-bending dichotomy of ice bearing capacity. Scaling
   exponents and asymptotes are verified numerically here, but a digitised
   overlay of a published capacity chart was **not** performed (no access to
   those data sets in this environment), so criterion (c) is met at the level of
   asymptotes/trends plus a classical closed-form anchor, not by curve-matching a
   figure from the literature.
4. **Reference length conventions.** Three lengths circulate in the literature:
   `(D/k)^(1/4)` (14.6 m for 1 m ice, the classical ice-sheet length),
   `(4D/k)^(1/4)` (20.7 m, the Hetenyi penetration length used by the closed
   form), and the two Pasternak layers. The module reports all of them explicitly
   (`l_ice`, `l_winkler`, `l_shear`, `l_soft`, `l_dominant`) to prevent silent
   factor-sqrt-2 mistakes.
5. **Reaction accounting subtlety.** In the weak form the shear term contributes
   nothing to a rigid vertical translation, so the *net* bed reaction is the
   Winkler integral only; `-G_s [w']` is an end traction that at a free end is
   balanced by the beam's internal shear. `foundation_reaction` returns both
   separately and the tests check equilibrium against the net value; adding the
   two double counts (about 10 % error in the pinned-end case).
6. **Degenerate foundation.** At exactly `G_s = 2 sqrt(D k)` the two roots merge
   and the split closed form loses all significant digits, so a double-root
   branch `w = P (1 + l|x|) exp(-l|x|) / (4 EI l^3)`, `M(0) = P/(4 l)` is used.
   The first version of that branch had the wrong sign for `M(0)`; it was caught
   by the FEM-vs-analytic plateau comparison (the sheet-length sweep at
   `G_s = 2 sqrt(D k)` returned a negative capacity), fixed, and is now covered by
   a test that checks the double-root penetration against the FE solution.
7. **Discretisation.** The moment converges at about 1 % with l_shear/12
   elements (second-derivative quantity) while the deflection is
   node-superconvergent; moment-based conclusions use meshes of at least 10
   elements per shear layer.
8. **Scope.** Bilateral (tension-capable) foundation, static loading, linear
   elasticity. Tensionless lift-off exists in `tensorfem.beam_foundation` but is
   not combined with the Pasternak bed here. No hydrodynamic coupling, no added
   mass.

## 7. Files

- `src/tensorfem/pasternak_beam.py` -- elements, closed form, block-Thomas
  solver, mode extraction, power-law helper.
- `src/tensorfem/ice_sheet_pasternak.py` -- ice properties, characteristic
  lengths, strip flexural capacity, crushing/foundering loads, failure mode,
  transition scan, plate cross-check.
- `tests/test_pasternak_beam.py` (33 tests), `tests/test_ice_sheet_pasternak.py`
  (21 tests).
- `benchmarks/pasternak_ice_sheet_bending.py` -- evidence driver.
- `artifacts/pasternak_ice_sheet/pasternak_ice_sheet_benchmark.json` -- every
  number above.
