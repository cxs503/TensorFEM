# Plate structure–acoustic radiation chain (FEM modes → Rayleigh / far-field → σ)

Mission node **P76-N08** asked for a vibration-acoustics baseline that links the
existing structural solver to the acoustic side of TensorFEM: FEM eigenpairs of a
baffled plate, the radiation efficiency `σ = P_rad/(ρ c S <v²>)`, an independent
far-field (boundary-integral) extrapolation, and the classical coincidence
asymptotics. This document records the model, the numbers, and what is *not*
achieved.

New code:

| path | content |
|---|---|
| `src/tensorfem/plate_acoustics.py` | plate configuration, coincidence frequency, sampled surface-velocity field, three acoustic power routes |
| `src/tensorfem/plate_radiation_chain.py` | FEM modal analysis, per-mode radiation efficiency, band average, asymptotics |
| `tests/test_plate_radiation_chain.py` | 25 tests (22 fast + 3 `slow`) |
| `scripts/plate_radiation_chain_benchmark.py` | staged evidence generator (writes `artifacts/plate_radiation_chain/`) |

Nothing in the existing modules was modified; the new code only consumes
`plate_vibration.solve_plate_vibration` and `acoustic_radiation._gauss_legendre`.

## 1. Case

Baffled rectangular steel panel radiating into air (all SI):

| quantity | symbol | value |
|---|---|---|
| length / width | `a`, `b` | 0.5 m / 0.5 m |
| thickness | `h` | 2 mm (`h/a = 0.004`) |
| Young / Poisson | `E`, `ν` | 210 GPa / 0.30 |
| plate density | `ρ_s` | 7850 kg/m³ |
| bending rigidity | `D` | 153.85 N·m |
| mass per area | `ρ_s h` | 15.7 kg/m² |
| longitudinal speed | `c_L = √(E/(ρ_s(1-ν²)))` | 5422 m/s |
| fluid | `ρ`, `c` | 1.225 kg/m³, 343 m/s |
| coincidence (engineering) | `f_c = c²/(1.8 c_L h)` | **6027.4 Hz** |
| coincidence (dispersion `c_B(f_c)=c`) | `c²/(2π)·√(ρ_s h/D)` | 5981.6 Hz (−0.77 %) |
| fundamental | `f_11` | 39.34 Hz |

## 2. Method

**Structure.** `fem_modal_analysis` calls the existing Q4 Mindlin plate solver
(selective reduced integration, consistent mass with rotary inertia) on a
structured `n×n` mesh, for `simple` and `clamped` edges. A mode vector is turned
into a `SurfaceVelocityField` by bilinear Q4 interpolation of the transverse DOFs
on a tensor Gauss grid and is rescaled to unit RMS velocity, so σ never depends on
the arbitrary eigenvector normalisation. Reference eigenvalues are the Kirchhoff
solution for simple support and Leissa's tabulated `λ` coefficients for the
clamped square plate.

**Acoustics — three independent routes for the same surface velocity.**

1. `rayleigh_power` — the 4-D Rayleigh surface integral
   `P = (ρω/4π)∬ v₁v₂ sin(kR₁₂)/R₁₂ dS₁dS₂` (the `sin` kernel is
   `Re[j e^{-jkR}]`; it vanishes on the diagonal, so there is no self-term).
2. `far_field_power` — far-field boundary-integral extrapolation:
   `p(r) = jωρ/(2πr) e^{jkr} A(θ,φ)` with the complex directivity
   `A = ∫_S v e^{-jk·r_s} dS`, hence `P = ρω²/(8π²c)∫|A|² dΩ`; `r` cancels. The
   half space is integrated with Gauss–Legendre nodes in `u = cos θ ∈ [0,1]`
   (grazing endpoint avoided) and a trapezoidal `φ`.
   *This is the "BEM far-field" leg.* The sibling repository TensorBEM 0.5.0 is a
   free-surface hydrodynamics BEM and has no acoustic half-space radiation
   module, so the half-space Green's-function extrapolation is implemented here
   instead of being imported; the cross-validation claim is therefore about the
   three *numerical formulations*, not about a second code base.
3. `spectral_power_sine_mode` — the wavenumber-domain form with the *closed-form*
   angular spectrum of a simply-supported sine mode,
   `A(κ) = V₀ I_x(κ_x) I_y(κ_y)`, `I_x = q(1-(-1)^m e^{-jκ_x a})/(q²-κ_x²)`
   (limit `-j a/2` at `κ_x = q`). No surface quadrature is involved, so this
   route stays usable far above coincidence where the other two cannot resolve
   the integrand any more.

**Convention.** `<v²>` is the time- and space-averaged mean square velocity, i.e.
`|V|²/2` for a complex amplitude `V` (`HARMONIC_TIME_FACTOR = 0.5` in the code).
In this convention a plane piston and a smooth plate mode both give `σ → 1` far
above coincidence, matching Leppington's asymptote.

**Band average.** `band_radiation_efficiency` collects the simply-supported modes
resonating inside a band (`f/√r … f√r`, default 1/3 octave), drives all of them
at the same frequency (equal modal energy, the Maidanik/reverberant assumption)
and averages their efficiencies.

## 3. Results

Artifacts: `artifacts/plate_radiation_chain/{modal,chain,crosscheck,acceptance}.json`
and four figures. `acceptance.json` is the machine-checkable criterion table.

### 3.1 Eigenfrequencies (criterion a)

| mode | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| FEM 24×24 (Hz) | 39.41 | 99.01 | 99.01 | 158.52 | 200.05 |
| Kirchhoff (Hz) | 39.34 | 98.34 | 98.34 | 157.35 | 196.69 |
| error | 0.19 % | 0.68 % | 0.68 % | 0.75 % | **1.71 %** |

Clamped 16×16: `1.00 %, 2.68 %, 2.68 %` against Leissa (CCCC ×`35.985, 73.394,
73.394`). The Q4 Mindlin element is stiff and converges only in the second order,
which is what sets the mesh requirement (the same panel at 12×12 gives 2.8 % /
7.3 % on modes 2 / 5).

### 3.2 Radiation efficiency and the acoustic cross-check (criteria b, c)

| field | `f` (Hz) | `σ` Rayleigh 4-D | `σ` far field | max gap | `σ` analytic sine |
|---|---|---|---|---|---|
| FEM (1,1) | 39.5 | 1.3633e-2 | 1.3633e-2 | 6e-16 | 1.3634e-2 (7.5e-5) |
| FEM (1,2) | 99.9 | 1.4879e-3 | 1.4879e-3 | 4e-16 | 1.4891e-3 (7.7e-4) |
| FEM (2,2) | 160.1 | 2.5407e-4 | 2.5407e-4 | 6e-16 | 2.5444e-4 (1.45e-3) |
| sine (9,15) at `f_c` | 6027.4 | 2.9567 | 2.9567 | 6e-16 | 2.9567 (1.5e-14) |

Route-to-route disagreement is at round-off level (`≲ 1e-15`), i.e. the two
acoustic formulations are algebraically equivalent and the discretisations are
converged in the tested range. FEM-driven σ differs from the analytical
sine-mode σ by ≤ 0.15 %, which is the discretisation error of the *mode shape*,
not of the acoustics.

### 3.3 Asymptotics (criterion d)

Fixed mode (1,1), frequencies 40–190 Hz (`f/f_c = 0.0066 … 0.0315`):

`σ = A (f/f_c)^p` with `p = 1.9446` (error 2.8 % against the classical 2) and
`A = 243.5`. This is the classical Maidanik dipole branch: below coincidence a
fixed mode radiates as `k² ∝ f²`.

High-frequency limit, mode (1,1) driven far above coincidence (closed-form route,
`n_θ×n_φ = 256×4096`):

| `f/f_c` | 2 | 4 | 10 |
|---|---|---|---|
| `σ` | 1.00081 | 1.00020 | **1.00003** |

Uniform piston check at `ka = 110`: `σ = 1.00085` (analytic value 1).

The **band average** does not follow `(f/f_c)²` — its measured exponent is 1.13
(samples 0.08–0.5 `f_c`). This is not a bug: in a band average the mode set
changes with frequency, so the dipole strength per mode falls as the order rises
and one power of `f` cancels. The `f²` law is a *fixed-mode* statement, and that
is the one the code verifies. See §5.

### 3.4 Coincidence frequency (criterion e) — honest partial failure

Band-averaged σ (`1/3` octave):

| `f/f_c` | 0.08 | 0.12 | 0.25 | 0.5 | 0.7 | 0.85 | 1.0 | 1.2 | 1.5 | 2.0 | 3.0 | 4.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `σ` | 0.015 | 0.016 | 0.036 | 0.089 | 0.227 | 0.94 | 1.92 | **2.58** | 1.81 | 1.44 | 1.16 | 1.19 |
| modes | 8 | 11 | 25 | 54 | 74 | 92 | 108 | 131 | 164 | 218 | 331 | 498 |

Narrowing the band to 1/12 octave and scanning 0.94–1.20 `f_c` puts the maximum at
`f/f_c = 1.0607` → **6.07 % above `c²/(1.8 c_L h)`**, i.e. the 5 % criterion is
**not met**. The 1/3-octave peak is worse (1.2 `f_c`, +20 %) because a wide band
averages modes whose bending wavenumber straddles the acoustic circle.

Why the peak sits above `f_c`: above coincidence every resonant mode obeys
`σ ≈ 1/√(1-f_c/f)`, which *diverges* at `f_c` and decays afterwards; a finite
panel with discrete modes can only sample that divergence from a distance. The
measured branch fits `σ = C/√(1-f₀/f)` with `f₀ = 1.053 f_c` (5.3 %, rms 1.5 %),
consistent with the peak position. Both estimators therefore agree that the
discrete band average locates coincidence a few percent high; the criterion's 5 %
is missed by ~1–1.3 percentage points, not by an order of magnitude, and it is
not covered up here.

### 3.5 Tests (criterion f)

`pytest tests/test_plate_radiation_chain.py` → 25 tests (22 fast, 3 `slow`).
Coverage: frequency references, closed-form transform vs quadrature, the three
routes against each other (three frequency decades incl. coincidence), the
`f²`/`σ→1` asymptotes, piston limit, band average and peak locator, FEM mode
normalisation, FEM-vs-analytical σ, validation errors. Full lane:

```
python -m pytest tests/test_plate_radiation_chain.py            # 25 passed
python -m pytest tests/test_plate_radiation_chain.py --fast     # 22 passed
```

## 4. Reproduction

```
python scripts/plate_radiation_chain_benchmark.py --stage modal      # ~1 min
python scripts/plate_radiation_chain_benchmark.py --stage chain      # ~1 min
python scripts/plate_radiation_chain_benchmark.py --stage crosscheck # ~1 min
python scripts/plate_radiation_chain_benchmark.py --stage figures
python scripts/plate_radiation_chain_benchmark.py --stage acceptance
```

## 5. Pitfalls, limitations, scope

* **Quadrature regimes.** The 4-D Rayleigh integral needs ~`λ/8` source spacing;
  it is converged at 40–48 Gauss points/side up to `ka ≈ 55` and loses 5 % at
  `ka ≈ 110` with 40 points (verified by doubling). The far-field angular route
  converges much later (`k a ≲ 110` with 192×2048). Both become unusable when the
  *sampled* surface grid cannot resolve the far-field phase: at `ka = 550` a
  32-point field returned `σ = 110` instead of 0.5 (peak convention) while the
  closed-form route was exact. Use the closed-form route for `ka ≳ 100`.
* **Band average ≠ single mode.** `σ ∝ (f/f_c)²` holds for a fixed mode; a
  resonant band average rises more slowly (measured exponent ≈ 1.1). Reporting
  the former as the band law would be a convention/physics error.
* **Harmonic factor.** With the peak-squared convention used by
  `acoustic_radiation.mode_mean_square_velocity` (`<v²> = V₀²/4`) all σ values
  here are exactly half of the rms-convention values. The repo's
  `leppington_radiation_efficiency` returns `1/√(1-(k_mn/k)²) ≤ 1` in the
  *same* peak-squared convention, so it is a factor of two above the exact
  Rayleigh result in the far-supersonic limit (`σ_peak = 0.5` for a smooth mode
  and for a piston). This is an audit finding for the existing module, left
  untouched here (`acoustic_radiation` is not part of this node) — the new chain
  uses the rms convention and states it explicitly.
  **Resolved 2026-11-08** — `docs/acoustic-radiation-convention-audit.md`: the
  root cause was not Leppington but the missing harmonic time average `1/2` in
  `acoustic_radiation.mode_mean_square_velocity` (`<v^2> = V0^2/4` instead of
  `V0^2/8`). With the denominator corrected the module's Rayleigh route
  reproduces the Leppington asymptote (1.0074 vs 1.0000 at 2 kHz, 5 kHz
  1.0012 vs 1.0000), so the chain's rms convention was the correct one and the
  published Leppington values never needed halving.
* **Thin-plate assumption.** `h/a = 0.004`; the analytical mode shapes and the
  `1/1.8` coincidence formula both assume Kirchhoff kinematics. Mindlin shear
  correction and rotary inertia are included in the FEM but not in the reference
  eigenvalues, which is part of the residual 0.2–1.7 % gap.
* **Clamped edges** are the worst case for the SRI Q4 element (2.7 % at 16×16);
  a finer mesh or a C¹ element would be needed for tight claims.
* **`solve_plate_vibration` mechanism detector.** At `h = 1 mm` the default
  `rigid_body_relative_tolerance = 1e-9` classifies the lowest elastic eigenvalue
  as a rigid-body mode and raises. The chain passes `0.0`; the underlying
  threshold is scale-dependent and should eventually be normalised by the
  bending-to-shear stiffness ratio.
* **No fluid loading.** The acoustic field is computed from a prescribed surface
  velocity (light-fluid assumption); added mass/damping of the air and any
  structure–acoustic feedback are outside this node.
* **Not covered:** point-force driven response with structural damping, infinite/
  orthotropic panels, and a genuinely independent second BEM code (see §2).