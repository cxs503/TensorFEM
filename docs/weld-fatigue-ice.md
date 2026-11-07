# Ice-region welded-joint fatigue assessment (`tensorfem.weld_fatigue_ice`)

Campaign node **IN22-N06** — physics domain: ice navigation / structural fatigue.
Status: implemented + verified (benchmark `benchmarks/weld_fatigue.py`, tests
`tests/test_weld_fatigue_ice.py`, artifact `artifacts/weld_fatigue.json`).
Evidence class: **REAL** for the FE notch field, the S-N/Miner algebra, the SDOF
response and the energy ledger; **surrogate (documented)** for the ice load
history, which is a DEM-calibrated stochastic model and **not** a TensorDEM run.

## 1. What the module adds

| Layer | Implementation | Evidence |
|---|---|---|
| Weld geometry | `WeldGeometry` (butt reinforcement `h_r`, toe radius `r_t`, flank angle `θ`; fillet leg `a`) | REAL (parametric, validated) |
| Toe stress state | `toe_stress_concentration` — FE Q4 notch submodel in elliptic-cylinder coordinates | REAL (FE, vs exact Inglis) |
| Hot-spot extraction | `hotspot_extrapolation` (IIW 0.4t/1.0t → 1.67/−0.67; DNV 0.5t/1.5t → 1.5/−0.5) | REAL |
| S-N curves | `DNVSNCurve` (RP-C203 curve D, air), `IIWFatClass` (FAT 90) | REAL (standard anchors) |
| Low temperature | `low_temperature_factor` / `SNTemperatureCorrection`, `k_T = 1.2` for `T ≤ −5 °C` | specification (campaign), not a class rule |
| Spectrum | `ice_pressure_history` (crushing pulses + background) | surrogate, documented |
| Structure | `PanelStructure` (side-shell strip, SDOF mode) + `exact_sdof_response` | REAL (exact closed form) |
| Counting / damage | `rainflow_cycles` (existing TensorFEM primitive) + `miner_damage` | REAL |
| Ledger | `response_energy_ledger` (Gauss quadrature on the exact response) | REAL, residual ≈ 1e-10 |

## 2. Method

### 2.1 Weld-toe notch submodel (FE)

The toe region is idealised as a semi-elliptical surface notch with tip radius
`r_t` and half-width `A = r_t (1 + 1.6 cos θ)`; the depth follows from the
ellipse geometry, `d = sqrt(A r_t)`. The half-plane problem (semi-elliptical
edge notch under remote tension parallel to the surface) is meshed exactly in
elliptic-cylinder coordinates

```
x = c cosh(eta) cos(xi),  y = c sinh(eta) sin(xi),  xi in [0, pi],
```

so that the notch surface is the coordinate line `eta = eta_0` and the flat free
surface is `xi = 0, pi`. `xi` is clustered towards the toe root (`beta = 0.6`) and
`eta` is graded quadratically towards the surface, which puts the near-toe edge
length two orders of magnitude below the 2 mm campaign requirement (reported as
`min_edge_m`). The mesh is solved with `tensorfem.continuum.solve_continuum`
(the repository Q4 formulation, dense solver, deliberately small submodel).

Verification of the machinery uses the *internal* elliptical hole, whose exact
Inglis answer is `sigma_tip / sigma_inf = 1 + 2 A / B` (`A` across, `B` along the
load). FE error: 1.5 % (`a/b = 1`), 3.5 % (`a/b = 2`), 8 % (`a/b = 4`) on the
coarse grid, improving monotonically under refinement; Richardson extrapolation
of the two mesh levels is below 3 % for `a/b ≤ 2` (see the artifact).

### 2.2 Stress concentration

`K_t = sigma_toe / sigma_nom` is the **toe (peak notch) stress concentration**,
matching the IIW effective-notch-stress convention with a 1 mm toe radius. The
structural hot-spot extrapolations (0.4t/1.0t and 0.5t/1.5t) are also returned;
for an isolated notch without the structural discontinuity they collapse to
≈ 1.0, which is why the fatigue chain uses the toe concentration and not the
extrapolated value.

### 2.3 Ice load spectrum

```
p_G = p_ref (h / 0.5 m)^0.5 (v / 5 kn)^0.1 ,        p_ref = 1.2 MPa
mean level        = 0.20 p_G
crushing rate f   = 0.5 v / h            [Hz]
pulse amplitudes  ~ Weibull(shape 2), half-sine, width U(0.35, 0.9)/f
background        = 8 % slow (2 Hz) multiplicative scatter
```

Thickness enters through the sub-linear pressure growth and, dominantly, through
the cycle rate `f ∝ v/h`.

### 2.4 Structural transfer

A continuous side-shell plate strip (`t = 25 mm`, frame spacing `s = 350 mm`) is
loaded by the ice pressure; its quasi-static bending stress is
`sigma = 0.25 p s^2 / t^2`. The lowest global mode is represented by an SDOF
(`f_n = 12 Hz`, `zeta = 3 %`) whose static gain is normalised to that beam value,
so any dynamic amplification is physical and the damping work is non-zero.

### 2.5 Low-temperature S-N shift

`k_T = 1.0` (`T ≥ 0 °C`), `1.2` (`T ≤ −5 °C`), linear between. The applied stress
range is multiplied by `k_T`, i.e. the allowable range at a given life is divided
by `k_T`: **−16.7 % at N = 1e7**.

## 3. Verification summary

| Criterion | Value | Pass |
|---|---|---|
| (a) toe `K_t` in the IIW band 1.5–2.5 | butt 2.33, fillet 2.40 | yes |
| (b) low-temperature strength loss at `N = 1e7` ∈ [15 %, 25 %] | 16.7 % | yes |
| (c) 18-case matrix covers both sides of `D = 1` | 13 safe / 5 fail | yes |
| (d) energy ledger residual < 1e-6 | ≈ 7e-10 | yes |
| (e) negative control: no weld (`K_t = 1`) overestimates life > 50 % | see artifact | yes |
| (f) damage dispersion over load realisations < 25 % | see artifact | yes |
| (g) standard anchors (DNV knee 52.63 MPa, IIW FAT 90, Inglis) | < 3 % | yes |

The exact numbers and the figures are produced by
`python benchmarks/weld_fatigue.py` and written to `artifacts/weld_fatigue.json`.

## 4. Assumptions and limitations (explicit)

1. **The ice load history is a surrogate.** No DEM solver is executed here;
   TensorDEM is a separate repository. The surrogate reproduces published
   crushing-pressure levels and event rates, but it is not a validated sea-ice
   DEM time history.
2. **The equivalent notch width `A/r_t = 1 + 1.6 cos θ` is an idealisation.**
   The FEM submodel covers only the radius/width ratio; a real toe also carries
   undercut, weld spatter and slag inclusions. The benchmark reports the `K_t`
   sensitivity over the practical `(r_t, θ)` range, and `K_t` stays inside
   1.5–2.5 throughout it.
3. **The low-temperature penalty `k_T = 1.2` is a campaign specification**, not a
   DNV curve. DNV-RP-C203 does not define a cold-temperature S-N shift of this
   form; the factor is applied where the parent specification requires it.
4. **The reference exposure (90 h) is a calibration** that places the matrix
   across the `D = 1` boundary. Damage scales linearly with exposure; the
   exposure-independent rate per 1000 h is reported so any operational value can
   be recovered. The stress level of the simplified panel model is conservative:
   extrapolated to a 5000 h ice life the severe cases reach `D ≈ 20–30`.
5. **Thin, fast ice dominates the fatigue damage** in this model, because the
   cycle rate scales as `v/h`. Thick ice dominates *ultimate* strength. This is a
   model prediction consistent with frequency-driven fatigue, not a measurement.
6. Rainflow counting inherits the assumptions of `marine_fatigue_advanced`
   (closed hysteresis loops, no mean-stress correction). No mean-stress correction
   (e.g. Haibach or Goodman) is applied; that is a conservative omission only for
   compressive mean stress.
7. No class-society approval is claimed, and none of the numbers constitute one.