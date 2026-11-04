# Wet ship-panel added mass: verification against Rayleigh's piston and the single-mode law

Status: **verification debt closed for the added-mass coefficient** (see §5 for what
remains open). Library: `TensorFEM`. Module: `tensorfem.hydro_added_mass`.
Benchmark: `benchmarks/wet_panel_added_mass.py` →
`artifacts/wet_panel_added_mass/validation_report.json`.

## 1. Why this case

`tensorfem.added_mass` adds a *prescribed* mass per unit area to a Q4 plate
element:

```
m_a = c_m * rho_w * L_char        (c_m ~ 1, L_char "characteristic length")
```

which is what lowers a wet panel's natural frequency. Until now the module had

* no test coverage at all, and
* **no reference for the coefficient itself**: `c_m` is user input, so the whole
  hydroelastic result depended on an unverified number.

That is a verification debt of exactly the kind the qualification matrix should
not carry silently (ship **vibration / noise** is one of the ten disciplines, and
panel wet frequencies drive fatigue-screen and comfort checks). This case closes
it with an independent hydrodynamic operator plus three anchors.

## 2. Method

### 2.1 Hydrodynamic operator (independent of `added_mass.py`)

The low-frequency (`k → 0`, in-phase) added mass of a plane surface embedded in
an infinite rigid baffle is the Rayleigh single-layer operator

```
M_ij = rho_w / (2 pi) * A_i A_j / |x_i - x_j|       i != j
M_ii = rho_w * A_i * a_i ,   a_i = sqrt(A_i / pi)
```

`M_ii` is the *exact* integral of the kernel over an equal-area disc,
`∫_disc dA/(2 pi r) = a`. Implemented in
`hydro_added_mass.dipole_added_mass_matrix`, row-chunked so a few thousand
panels fit in memory.

For a simply-supported panel mode `phi = sin(m pi x/lx) sin(n pi y/ly)` the
modal added-mass ratio is

```
mu = phi^T M phi / (rho_s t lx ly / 4)        (∫sin^2 = L/2 exactly)
```

and the exact single-mode in-phase frequency law is
`f_wet / f_dry = 1/sqrt(1 + mu)`.

### 2.2 Structural side

Q4 Mindlin (SRI) plate elements, `tensorfem.plate.q4_mindlin_stiffness`;
transverse-DOF consistent mass from `tensorfem.added_mass` (same shape
functions as the added mass). Rotational DOFs are statically condensed
(`K_cond = K_ww - K_wr K_rr^-1 K_rw`) so that the eigenproblem lives on the same
`w` space as the added-mass matrix. Simply supported edges (`w = 0`),
rotations free.

### 2.3 Verification anchors

| # | Reference | Gate |
|---|---|---|
| A1 | Rayleigh baffled-piston radiation mass `(8/3) rho a^3` (Rayleigh 1945 §302) | < 3 % for ≥ 64 cells/side |
| A2 | Kirchhoff Navier dry frequencies `pi^2 (m^2/lx^2+n^2/ly^2) sqrt(D/rho_s t)` | < 3 % per mode |
| A3 | Exact single-mode law `f_wet/f_dry = 1/sqrt(1+mu)` with `m_a` from A1-validated operator fed into `assemble_added_mass_consistent` | < 3 % |
| A4 | `m_a = c_m rho_w L_char` heuristic vs the exact operator (informative) | no gate, reports the gap |

A3 is a *consistency* anchor: it proves the element assembly carries exactly the
generalised mass the hydrodynamic operator asks for (partition of unity, correct
`w`-DOF placement, lumped/consistent parity). The physical validity of `mu`
itself rests on A1, where the operator is compared with an exact analytic
result.

## 3. Case: side-shell strake between frames

| Parameter | Value |
|---|---|
| panel `lx × ly × t` | 1.2 m × 0.8 m × 0.008 m |
| steel `E`, `nu`, `rho_s` | 210 GPa, 0.3, 7850 kg/m³ |
| water `rho_w` | 1025 kg/m³ |
| structural mass per area `rho_s t` | 62.8 kg/m² |
| boundary | simply supported, mode (1,1) |

All numbers below are read programmatically from
`artifacts/wet_panel_added_mass/validation_report.json` (no hand-copied values).

## 4. Results

### A1 — baffled piston, `a = 0.5 m`, `rho = 1025 kg/m³`, exact `M = 341.667 kg`

| cells/side | panels | `sum(M)` [kg] | rel. error | area error | area-normalised error |
|---|---|---|---|---|---|
| 16 | 208 | 357.395 | +4.60 % | +3.45 % | +1.11 % |
| 32 | 812 | 345.382 | +1.09 % | +0.96 % | +0.12 % |
| 64 | 3228 | 342.748 | +0.32 % | +0.34 % | +0.03 % |
| 96 | 7232 | 340.773 | −0.26 % | −0.09 % | — |
| 128 | 12892 | 342.280 | +0.18 % | +0.19 % | — |

The raw error tracks the staircase *area* error almost one-to-one, i.e. the
kernel and its normalisation are exact and the residual is purely geometric
truncation of the disc. Once the discretised area is divided out the operator is
converged to 0.03 % by 64 cells/side. Convergence is monotone until ~1 % and then
oscillates at the staircase level (±0.3 %), which is why the gate is stated on
`|error| < 3 %` at ≥ 64 cells rather than on strict monotonicity.

### A2 — dry panel frequencies

| mode | Navier [Hz] | FE 24×16 [Hz] | rel. error |
|---|---|---|---|
| (1,1) | 44.391 | 44.397 | +0.01 % |
| (2,1) | 85.367 | 85.587 | +0.26 % |
| (1,2) | 136.587 | 138.696 | +1.54 % |

Mesh sequence for the highest mode (1,2): +13.3 % (8×6) → +2.9 % (16×12) → +1.5 % (24×16).
The 3 % gate is therefore applied on the finest mesh; on the coarse 8×6 mesh the
second half-wave (0.4 m over 6 elements) is simply under-resolved, which the table
above makes explicit instead of hiding behind an averaged number.

### A3 — wet ratio and the hydrodynamic coefficient

| quantity | value |
|---|---|
| `mu` (modal added-mass ratio, 60×40 panels) | 6.465 |
| `mu` (20×14) / (30×20) / (45×30) / (60×40) | 6.450 / 6.455 / 6.461 / 6.465 (0.24 % drift, monotone) |
| equivalent uniform `m_a = mu rho_s t` | 406.0 kg/m² |
| exact `f_wet/f_dry = 1/sqrt(1+mu)` | 0.36600 |
| FE `f_dry` → `f_wet` (24×16) | 44.397 Hz → 16.250 Hz, ratio 0.36600 → error 0.00 % |

So a 1.2 × 0.8 × 8 mm steel strake in water vibrates at ~16 Hz instead of
~44 Hz: the hydrodynamic mass is 6.5× the structural mass, which is why wet
panel modes cannot be screened with dry frequencies.

### A4 — the `c_m rho_w L_char` heuristic is off by a factor of ~2

| `L_char` | `m_a` at `c_m = 1` | `mu` | error vs exact | calibrated `c_m*` |
|---|---|---|---|---|
| `ly` = 0.8 m | 820 kg/m² | 13.06 | **+102 %** | 0.495 |
| `lx` = 1.2 m | 1230 kg/m² | 19.59 | +203 % | 0.330 |
| `2 lx ly/(lx+ly)` = 0.96 m | 984 kg/m² | 15.67 | +142 % | 0.413 |
| `t` = 0.008 m | 8.2 kg/m² | 0.131 | −98 % | 49.5 |

**Conclusion for `added_mass.py`:** with `c_m = 1` and any geometric length the
heuristic over-predicts the added mass by a factor 2–3 for a 1.5:1 panel; with
`L_char = t` it under-predicts by ~50×. The calibrated coefficients
(`c_m* ≈ 0.33–0.50`) are aspect-ratio dependent, so `c_m` must not be treated as
a universal constant. Where a coefficient is needed, use
`hydro_added_mass.panel_added_mass_ratio(...)` instead of guessing `c_m`.

## 5. Caveats and scope

* **In-phase limit only.** (The finite-`ka` continuation, including the radiation
  damping this bullet says is missing, is now implemented and verified against
  Rayleigh's piston closed form in
  `docs/piston-radiation-impedance-verification.md`.) The operator is the `k → 0`
  Rayleigh integral. It is
  valid when the acoustic wavelength is long compared with the panel
  (`k a << 1`); here `f = 16 Hz` in water gives `k a ≈ 0.04`, so the assumption is
  comfortable. At higher modes / higher frequency the full
  `exp(-j k r)/(2 pi r)` kernel with the radiation damping term must be used.
* **One-sided wetting.** Fluid on one side, infinite rigid baffle, no free
  surface, no compressibility, no mean flow. Two-sided wetting doubles `m_a`.
* **No added damping.** Only the reactive (mass) part is modelled; radiation
  damping is not included, so wet resonances are undamped here.
* **Staircase discretisation.** The disc/panel meshes are Cartesian, so the
  geometric error is `O(h)`; a boundary-fitted panel mesh would converge faster.
* **Kirchhoff Navier reference for A2.** Shear deformation is negligible at
  `t/lx = 0.0067`, which is why the thin-plate reference is admissible; the
  residual 1.5 % on mode (1,2) is Q4 Mindlin interpolation, not physics.
* **A3 is consistency, not physics.** It cannot fail for a wrong-but-consistent
  coefficient; the physical anchor is A1.

## 6. Error judgement (key quantities vs benchmark)

| Quantity | Benchmark | Error | < 3 % ? |
|---|---|---|---|
| Piston added mass, 64 cells/side | `(8/3) rho a^3` (exact) | **0.32 %** | **yes** |
| Piston added mass, area-normalised | `(8/3) rho a^3 / (pi a^2)` | 0.03 % | yes |
| Dry `f_11` (24×16) | Navier | 0.01 % | yes |
| Dry `f_21` (24×16) | Navier | 0.26 % | yes |
| Dry `f_12` (24×16) | Navier | 1.54 % | yes |
| Wet/dry ratio | `1/sqrt(1+mu)` | 0.00 % | yes (consistency) |
| Modal `mu` vs panel-mesh refinement | 30×20 → 60×40 | 0.16 % | yes (self-convergence) |
| Dry `f_12` on the coarse 8×6 mesh | Navier | 13.3 % | no (under-resolved, excluded by the gate) |
| `m_a = rho_w ly` heuristic | dipole operator | **+102 %** | **no — this is the finding** |

## 7. Reproduction

```
PYTHONPATH=src python benchmarks/wet_panel_added_mass.py
PYTHONPATH=src python -m pytest tests/test_hydro_added_mass.py -q   # 12 passed
```

Artifacts: `artifacts/wet_panel_added_mass/validation_report.json`,
`piston_and_wet_ratio.png` (A1 convergence + A3 ratio), `panel_dipole_field.png`
(hydrodynamic load of mode (1,1) over the panel).