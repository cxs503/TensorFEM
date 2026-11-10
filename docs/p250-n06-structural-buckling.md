# p250-N06 -- plate/shell buckling and postbuckling face

**Anchor.** `fem_plate_postbuckling` -- driver `benchmarks/verified/fem_plate_postbuckling/run.py`,
machine contract `gates_verified.json` (schema `tensorfem.verified-gates.v1`), raw numbers
`result.json`. Evidence class: **REAL** (finite element + arc-length
continuation + closed-form oracles), no MOCK/STUB anywhere on the path.

**Verdict.** All 6 gates pass
(`passed = True`); failed gates: `[]`. The
module `src/tensorfem/von_karman_plate_riks.py` is new; tests
`tests/test_von_karman_plate_riks.py` (24 cases, ~22 s).

## 1. Case

Plate field of the campaign stiffened panel with the stiffener removed, so that
every closed-form oracle applies verbatim:

| quantity | value |
|---|---|
| loaded length a | 400 mm |
| supported width b | 400 mm |
| thickness t | 6.0 mm |
| E, nu | 206 GPa, 0.3 |
| yield stress | 355 MPa |
| plate slenderness beta = (b/t) sqrt(sigma_y/E) | 2.768 |
| bending rigidity D | 4074.73 N m |
| Timoshenko sigma_cr (k = 4) | 167.57 MPa |

Edges: simply supported (`w = 0` and zero slope *along* each straight edge);
in plane `u = 0` on `x = 0`, uniform traction on `x = a`, stress-free unloaded
edges, one pinned corner. The one-mode reference below assumes a single buckling
mode and the particular Airy stress function only (zero *mean* membrane stress in
the unloaded direction).

## 2. Method

* **Element.** New rectangular von Karman plate element, six DOFs per node
  `(u, v, w, w_,x, w_,y, w_,xy)`: bending from the Bogner--Fox--Schmit bicubic
  Hermite (C1, 3x3 Gauss), membrane from the bilinear Q4 on the same corners,
  coupling through `eps = B_L u + (w_t^2 - w0^2)/2` style von Karman strains so a
  seeded imperfection `w0` is a **stress-free** initial configuration. Analytic
  tangent (material `B^T C B` plus initial stress `G^T S G`).
* **Bifurcation.** `K_G` is taken as the *difference* `K_T(u_pre) - K_T(0)` of the
  assembled nonlinear tangent at unit reference line load (prebuckling state from
  the same element, not an injected stress field); the pencil
  `(K + lambda K_G) phi = 0` is mapped to a standard symmetric problem with the
  Cholesky factor of `K` and solved with LAPACK; each reported mode carries its
  pencil residual.
* **Continuation.** Crisfield arc length from `tensorfem.arc_length` (reused, not
  reimplemented) on the physical DOF vector, with a per-DOF scaling that makes one
  arc step mean one step of bending amplitude. The trace starts from a Newton
  equilibrium at `N/N_cr = 0.9` (preload offset), so the arc budget is spent on
  the postbuckling branch instead of the boring prebuckling compression.
* **Oracles.** Timoshenko `sigma_cr = k pi^2 D/(b^2 t)`,
  `k = min_m (m b/a + a/(m b))^2`; one-mode von Karman
  `N/N_cr = (1 - w0/A) + c (A/t)^2`,
  `c = 3(1-nu^2)(alpha^4+beta^4)/(4(alpha^2+beta^2)^2)` = 0.34125
  for the square plate; DNV-RP-C201 plate/column mode families for the panel.

## 3. Results

### 3.1 A1 -- linear bifurcation vs Timoshenko (gate < 3 %)

| a/b | m | k | sigma_cr closed form [MPa] | mesh | sigma_cr FE [MPa] | rel. error | MAC(mode) | pencil residual |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 4.0 | 167.57 | 4x4 | 167.610 | +2.608e-02 % | 1.0000 | 4.9e-15 |
| 1 | 1 | 4.0 | 167.57 | 6x6 | 167.575 | +5.189e-03 % | 1.0000 | 1.6e-14 |
| 1 | 1 | 4.0 | 167.57 | 8x8 | 167.569 | +1.629e-03 % | 1.0000 | 4.4e-14 |
| 2 | 2 | 4.0 | 167.57 | 8x4 | 167.610 | +2.608e-02 % | 1.0000 | 5.5e-15 |
| 2 | 2 | 4.0 | 167.57 | 12x6 | 167.575 | +5.189e-03 % | 1.0000 | 2.4e-14 |
| 2 | 2 | 4.0 | 167.57 | 16x8 | 167.569 | +1.628e-03 % | 1.0000 | 5.2e-14 |
| 3 | 3 | 4.0 | 167.57 | 12x4 | 167.610 | +2.608e-02 % | 1.0000 | 6.2e-15 |
| 3 | 3 | 4.0 | 167.57 | 18x6 | 167.575 | +5.189e-03 % | 1.0000 | 2.4e-14 |
| 3 | 3 | 4.0 | 167.57 | 24x8 | 167.569 | +1.629e-03 % | 1.0000 | 5.9e-14 |

Max |relative error| over all aspect ratios and meshes: **0.0261 %**
-- about two orders of magnitude inside the 3 % gate, and the mode MAC against
the closed-form shape is 1.0000 because the bicubic Hermite resolves the sine
mode essentially exactly. Note the convergence is *from above* at a fixed
element size: 4x4 gives +2.6e-4, 8x8 gives +1.6e-5.

### 3.2 A2 -- postbuckling path vs one-mode von Karman--Koiter (gate < 5 %)

Near-perfect plate, `w0 = t/1000` (seeded so that
the bifurcation is resolvable in load control). Load factor = `N/N_cr` against the
**total** amplitude `A/t`, ten stations per mesh:

| mesh | A/t=0.10 | A/t=0.15 | A/t=0.20 | A/t=0.25 | A/t=0.30 | A/t=0.35 | A/t=0.40 | A/t=0.45 | A/t=0.50 | A/t=0.55 |
|---|---|---|---|---|---|---|---|---|---|---|
| 8x8 | -0.13 % | -0.23 % | -0.40 % | -0.60 % | -0.90 % | -1.17 % | -1.57 % | -1.91 % | -2.35 % | -2.85 % |
| 12x12 | -0.15 % | -0.28 % | -0.47 % | -0.72 % | -1.06 % | -1.40 % | -1.85 % | -2.29 % | -2.83 % | -3.36 % |

Max |relative error| over the 10 stations and both meshes: **3.36 %**
(gate 5 %), and it is *not* a discretisation error: the two meshes differ by
about 0.5 percentage points at `A/t = 0.55` and by 1.5 points at `A/t = 1`.

**The one-mode reduction is an asymptotic in the amplitude, and the anchor says so
with numbers instead of stopping the comparison at a convenient amplitude:**

| A/t | FE load factor 8x8 | FE load factor 12x12 | one-mode Koiter | deviation (12x12) |
|---|---|---|---|---|
| 0.6 | 1.0841 | 1.0767 | 1.1212 | -3.97 % |
| 0.8 | 1.1488 | 1.1365 | 1.2172 | -6.62 % |
| 1.0 | 1.2315 | 1.2114 | 1.3402 | -9.61 % |

The deflection field itself stays single-mode: MAC of the FE deflection against
`sin(m pi x/a) sin(n pi y/b)` as the amplitude grows:

| A/t | N/N_cr | MAC vs sin.sin (whole field) | MAC (centre line) |
|---|---|---|---|
| 0.21 | 1.006 | 1.0000 | 1.0000 |
| 0.41 | 1.037 | 1.0000 | 1.0000 |
| 0.54 | 1.066 | 0.9999 | 0.9999 |
| 0.79 | 1.143 | 0.9997 | 0.9998 |
| 1.01 | 1.237 | 0.9993 | 0.9994 |

so the residual difference cannot be blamed on the mode *shape*.
The remaining candidates are the membrane description -- the finite element
carries the dead-load (uniform traction) edge with a free in-plane warp and a
full redistribution of `N_x`, while the one-mode energy assumed
`eps_bar = N/(E t)` with zero mean membrane stress in the unloaded direction --
and the higher harmonics of the stress function. This is stated as an open
question, not resolved by assertion; the concrete next check is a rigid-platen
(uniform end-shortening) companion run and a two-mode reduction.

### 3.3 A3 -- imperfection sensitivity vs Koiter (gate < 10 %)

Load factor at the fixed total amplitude `A/t = 0.25`:

| w0 | w0 [mm] | N/N_cr FE | N/N_cr Koiter | rel. error |
|---|---|---|---|---|
| t/500 | 0.012 | 1.0070 | 1.0133 | -0.63 % |
| t/200 | 0.030 | 0.9949 | 1.0013 | -0.64 % |
| t/100 | 0.060 | 0.9748 | 0.9813 | -0.67 % |

Max |relative error| **0.67 %** (gate 10 %).
The physics is the honest plate result, not a knockdown: the plate branch is
*stable*, so there is no limit point to knock down, and the imperfection acts as
a compliance (the same load is reached at a larger amplitude). The sign of the
effect is monotone in `w0` in both the FE and the one-mode solution.

### 3.4 A4 -- stiffened-panel buckling mode vs DNV-RP-C201 (gate MAC > 0.9)

Panel unit cell of `tensorfem.stiffened_panel_buckling` (span 800 mm,
spacing 400 mm, plate 6.0 mm,
tee 150x10 +
50x10 mm, 8x4 mesh).

| DNV mode family | MAC of the FE eigenmode |
|---|---|
| plate panel, m = 1..6 (quarter wave across the half bay) | 1: 0.0000, 2: 0.0000, 3: 0.9803, 4: 0.0000, 5: 0.0000, 6: 0.0000 |
| stiffener column (uniform across the bay) | 2.06e-08 |

* Best plate-family member: `m = 3`, MAC = **0.9803**
  (gate 0.9); column mode MAC = 2.1e-08, i.e. orthogonal.
* FE critical stress 319.9 MPa.
* DNV-RP-C201 closed forms for the same scantlings: plate panel
  167.6 MPa (k = 4),
  elastic Euler stiffener column 4813 MPa
  capped by yield to 355.0 MPa.
  DNV therefore classifies the panel as **plate-governed**,
  the FE mode is **plate-dominated**, and the two agree:
  `True`.

The FE cell value sits above the DNV simply-supported plate value because the
unit cell reflects the *rotational* restraint of the stiffener line (the
stiffener is a nodal line for the plate mode), which raises the coefficient; the
gate here is mode-family consistency, the stress comparison is context.

### 3.5 A5 -- negative controls

* Zero preload: the assembled tangent is positive definite, smallest eigenvalue
  0.818 N/m over
  408 free DOFs -- no zero-energy (hourglass)
  mode and therefore no spurious bifurcation. `no_zero_energy_mode = True`.
* Tensile reference load: the mapped eigenvalue spectrum is non-positive
  (largest 7.43e-23, below the
  relative floor), so no positive buckling factor exists:
  `tensile_has_no_buckling_factor = True`.

## 4. Figures

* `benchmarks/verified/fem_plate_postbuckling/buckling_mode.png` -- buckling
  mode field (contour) and the centre-line profile against `sin(pi x/a)`.
* `benchmarks/verified/fem_plate_postbuckling/postbuckling_path.png` -- load
  factor against total amplitude for two meshes with the one-mode solution, plus
  the deviation from it versus amplitude with the gate range shaded.
* `benchmarks/verified/fem_plate_postbuckling/imperfection_sensitivity.png` --
  load-amplitude curves for `w0 = t/500, t/200, t/100` (FE solid, Koiter dashed)
  and the sensitivity at fixed amplitude.

## 5. Error judgement (does every key quantity stay below 3 %?)

| criterion | quantity | value | gate | passed |
|---|---|---|---|---|
| A1 | max |sigma_cr_FE / sigma_cr_Timoshenko - 1| over a/b = 1,2,3 | 0.0002608 | <= 0.03 | True |
| A2 | max |N_FE / N_koiter - 1| at 10 path stations (A/t = 0.10..0.55) | 0.03356 | <= 0.05 | True |
| A3 | max |N_FE / N_koiter - 1| over w0 = t/500, t/200, t/100 | 0.006662 | <= 0.1 | True |
| A4 | MAC of the panel eigenmode against its DNV-RP-C201 mode family | 0.9803 | >= 0.9 | True |
| A5a | min eigenvalue of the zero-load tangent | 0.8184 | >= 0.0 | True |
| A5b | count of positive buckling factors under tension | 0 | <= 0.0 | True |

The campaign's blanket 3 % accuracy standard cannot be applied to every gate:
A2 and A3 are graded against a **one-mode asymptotic** whose own truncation error
grows with amplitude, so their gates are 5 % and 10 % by construction of the
campaign mandate. The two quantities that *are* graded against an exact closed
form (A1 linear bifurcation, and the element's own bending response, which is
checked in the test suite against the Navier value 0.00406 q b^4/D within 4 %
including the lumped-load error) sit at 0.0261 %
and ~2.3 % respectively -- inside 3 %.

## 6. Limitations and residual risk

* Small-rotation von Karman plate element (BFS bending + Q4 membrane); finite rotations, yielding and stiffener web local buckling are outside the model.
* Dead-load (uniform traction) in-plane loading with stress-free unloaded edges; a rigid end platen would raise the postbuckling load.
* The one-mode Koiter reference is an asymptotic in the amplitude, so the A2 gate is graded inside A/t <= 0.55; the deviation beyond that range is reported as a measurement, not hidden.
* The panel MAC comparison (A4) uses the transverse nodal field only, which is free of any rotation sign convention.
* The element is a *small-rotation* von Karman plate. At `A/t = 1` the rotations
  are ~2.5 degrees, so the model is used inside its stated range; nothing here
  says anything about `A/t > 2`, plasticity, or stiffener web local buckling.
* The arc-length trace needs a resolvable bifurcation: at `w0 = t/10000` the
  continuation stays on the (unstable) flat branch after `N_cr`, which is a
  correct equilibrium of the near-perfect plate but not the postbuckling branch.
  `run.py` therefore seeds `w0 = t/1000` and says so.
* BLAS threading: every step is a chain of small dense factorisations; on a
  many-core host an oversubscribed BLAS makes them ~40x slower. `run.py` pins
  `OPENBLAS_NUM_THREADS=1` before importing numpy (measured: 9.0 s vs ~6 min for
  the A1 sweep).

## 7. Reproduce

```bash
PYTHONPATH=src python benchmarks/verified/fem_plate_postbuckling/run.py      # ~2 min
PYTHONPATH=src python benchmarks/verified/fem_plate_postbuckling/render_doc.py
PYTHONPATH=src python -m pytest tests/test_von_karman_plate_riks.py -q        # 24 cases, ~22 s
```
