# Corrosion-ice time-dependent ultimate strength of polar hull plating

Qualification of `tensorfem.marine_corrosion_ice_ultimate`: a stochastic pitting
field, a field-resolved plate ultimate strength, a Paik-type empirical anchor, an
ice-pressure demand model, and the inverse inspection criterion for the ice belt
of an ice-class bulk carrier.

Run with:

```bash
PYTHONPATH=src python -m pytest -q tests/test_marine_corrosion_ice_ultimate.py
PYTHONPATH=src python scripts/run_w10n08_corrosion_ice_ultimate.py          # ~10 min
PYTHONPATH=src python scripts/report_w10n08.py                              # report body
```

Evidence lands in `artifacts/w10n08_corrosion_ice_ultimate/assessment.json` and
`docs/assets/marine_corrosion_ice/`.

## What is computed

1. **Pitting field** (`generate_pitting_field`).  A Gaussian covariance random
   field (FFT spectral synthesis, correlation length an explicit input) modulates
   the intensity of a Cox pit process; pit diameters are log-normal, pit depth
   factors Weibull (`k = 1.2`, unit mean) and initiation times uniform inside the
   coating-breakdown window.  The number of pits is *inverted* from
   `covered = 1 - exp(-A_p / A)` so that the realised DOP equals the requested
   intensity instead of saturating through disc overlap.  Pits are rasterised on
   a `diameter/4` lattice and aggregated to the eigen cells.  Two depth models:
   `corrosion_law_volume` (the corrosion law fixes the mean wastage over the whole
   surface and the census fixes how localised it is, so the mean pit depth is
   `wastage / pitted fraction`) and `depth_ratio` (mean pit depth pinned to a
   fraction of the plate thickness, used for the Paik parameter range).
2. **Ultimate strength** (`plate_ultimate_strength`).  Each eigen cell carries the
   Voigt-average bending rigidity (`mean(t^3)`, *not* the cube of the mean
   thickness), and the Fourier-Galerkin projection on
   `sin(m pi x / a) sin(n pi y / b)` (`m <= 8`, `n <= 3`) assembles the
   variable-thickness generalised eigenproblem `K_b v = N K_g v`.  The inelastic
   collapse follows the classical effective-width law with
   `beta^2 = (4 pi^2 / (12 (1 - nu^2))) sigma_y / sigma_e`.  The net-section
   squash limit is checked as a second mechanism and the thinnest patch is
   reported as a diagnostic (with the patch size at which local buckling becomes
   critical).
3. **Empirical anchor** (`paik_pitted_reduction`).  `R(DOP) = 1 - a DOP^b` with a
   default `(a, b) = (0.33, 1)` (about 10 % loss at `DOP = 0.3`).  This is a
   secondary-literature transcription: the primary table of Paik et al. (2003)
   is not available offline, which is why `anchor_sensitivity` reports the
   `a in [0.20, 0.45]`, `b in {0.75, 1}` band and why the anchor gate is an
   *internal consistency* gate, not a validation against the published table.
4. **Ice demand** (`ice_demand`).  `F = p(A) A_contact` from the pressure-area law
   `p = p_ref (A/A_ref)^-beta`, transferred as in-plane membrane action along two
   panel edges: `sigma_ice = F / (2 b t)`.  The clamped circular plate
   large-deflection Rayleigh-Ritz solution is kept as a *local screening* only:
   at 2.5-3.5 MPa its `w/t` reaches 1.0-1.25 and its trial-family spread is about
   30 %, so it cannot carry the margin.
5. **Low temperature**.  EH36/DH36 yield stress with a linear temperature shift
   (flagged as an estimate); brittle-fracture screening is deliberately out of
   scope.
6. **Inverse problem**.  `allowable_service_life` solves `sigma_u^p05(t) =
   gamma sigma_ice^p95(t)`; `allowable_pitting_depth` bisects on the mean pit
   depth at the design age (well posed for both depth models) and reports the
   equivalent mean pitting rate; `inspection_schedule` turns both into an
   inspection table capped at five years.

## Acceptance gates

| Gate | Threshold | Where |
|---|---|---|
| (a) field solver vs Paik-type anchor, five DOP levels | < 10 % | `criteria.a_paik_anchor` |
| (b) uniform wastage mass loss, annual midpoint vs closed form | < 5 % | `criteria.b_uniform_mass_loss` |
| (c) DOP = 0 vs the existing plate-buckling node, same geometry/load | < 2 % | `criteria.c_degeneracy_dop_zero` |
| (d) `U(t)` monotone non-decreasing, 100 random fields | 0 violations | `criteria.d_monotonicity` |
| (e) P50 strength, 500 vs 1000 realisations | < 1 % | `criteria.e_monte_carlo_convergence` |
| extra: eigen-grid convergence of one pitted realisation | < 3 % | `grid_convergence` |

`tests/test_marine_corrosion_ice_ultimate.py` contains 46 tests; the heavy
pairs (500/1000 samples) are executed by the study script and stored in the
artifact, the test suite runs a 40/80 pair on a coarser realisation.

## Known limitations

* The Paik anchor coefficients are unverified against the primary source; the
  gate proves consistency with the *implemented* anchor only.
* The field-resolved elastic buckling stress is a modal Galerkin quantity
  (discretisation uncertainty 1-2 %); local perturbations inside a pit are not
  resolved and a buckling patch is allowed to shed load, which is why the
  first-patch stress is reported but never allowed to govern.
* The ice pressure is a place-holder calibrated so the intact panel sits at
  `U = 0.55`; the demand model is statics (force spread), not a shell solution.
* Not covered: residual stress, stiffener tripping, brittle fracture,
  corrosion-fatigue interaction, cyclic ice statistics and hydrodynamic
  amplification, weld heat-affected zones, class-rule checks.
* `scipy.linalg`'s generalised eigensolver aborts the interpreter in the
  qualification environment, so the modal route reduces to a symmetric problem
  with a numpy Cholesky factorisation and the finite-difference route (sparse
  shift-invert) is verification only.
## Measured results (generated from `artifacts/w10n08_corrosion_ice_ultimate/assessment.json`)

### Acceptance gates

| Gate | Threshold | Baseline | Severe |
|---|---|---|---|
| (a) field solver vs Paik-type anchor | < 10 % | 1.03 % | 1.03 % |
| (b) mass loss, midpoint sum vs closed form | < 5 % | 0.0246 % | 0.0246 % |
| (c) DOP = 0 vs sibling plate-buckling node | < 2 % | 0.205 % | 0.205 % |
| (c') same, vs Navier closed form | < 2 % | 1.22e-13 % | 1.22e-13 % |
| (d) U(t) monotone, 100 random fields | 0 violations | 0 | 0 |
| (e) P50 strength, 500 vs 1000 realisations | < 1 % | 0.0075 % | 0.0303 % |
| eigen-grid convergence (coarsest vs finest) | < 3 % | 0.027 % | 1.110 % |

### Paik cross-check (mean pit depth = 0.5 t at 25 years)

| DOP requested | DOP realised | sigma_e / sigma_e,0 | R field | R anchor | deviation |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 1.0000 | 1.0000 | 1.0000 | 0.00 % |
| 0.075 | 0.083 | 0.9522 | 0.9823 | 0.9725 | 1.02 % |
| 0.150 | 0.143 | 0.9016 | 0.9628 | 0.9529 | 1.03 % |
| 0.225 | 0.222 | 0.8264 | 0.9319 | 0.9266 | 0.57 % |
| 0.300 | 0.289 | 0.7576 | 0.9016 | 0.9045 | 0.32 % |

### Baseline decadal history (Melchers wastage, 100 realisations)

| years | DOP p50 | t p50 [mm] | sigma_u p50 [MPa] | sigma_u p05 [MPa] | sigma_ice p50 [MPa] | U p50 | R p50 |
|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 15.00 | 262.4 | 262.4 | 144.3 | 0.550 | 1.000 |
| 5 | 0.073 | 14.84 | 260.5 | 259.8 | 145.9 | 0.560 | 0.993 |
| 10 | 0.134 | 14.78 | 259.5 | 258.4 | 146.4 | 0.564 | 0.989 |
| 15 | 0.189 | 14.74 | 258.8 | 257.6 | 146.8 | 0.567 | 0.986 |
| 20 | 0.189 | 14.70 | 258.2 | 256.7 | 147.2 | 0.570 | 0.984 |
| 25 | 0.189 | 14.67 | 257.7 | 255.8 | 147.6 | 0.573 | 0.982 |
| 30 | 0.189 | 14.64 | 257.2 | 255.1 | 147.9 | 0.575 | 0.980 |
| 40 | 0.189 | 14.58 | 256.4 | 253.8 | 148.5 | 0.579 | 0.977 |
| 50 | 0.189 | 14.53 | 255.7 | 252.7 | 149.0 | 0.583 | 0.974 |

### Severe scenario (deep pitting, mean pit depth 0.55 t at 25 years)

| years | DOP p50 | sigma_u p50 [MPa] | sigma_u p05 [MPa] | sigma_ice p50 [MPa] | U p50 | U p95 | R p50 |
|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 262.4 | 262.4 | 144.3 | 0.550 | 0.550 | 1.000 |
| 5 | 0.091 | 259.2 | 258.1 | 147.0 | 0.567 | 0.571 | 0.988 |
| 10 | 0.168 | 253.2 | 247.8 | 151.8 | 0.599 | 0.614 | 0.965 |
| 15 | 0.233 | 245.6 | 237.8 | 157.5 | 0.642 | 0.665 | 0.936 |
| 20 | 0.233 | 241.4 | 231.7 | 161.7 | 0.671 | 0.699 | 0.920 |
| 25 | 0.233 | 239.4 | 228.4 | 164.5 | 0.688 | 0.721 | 0.913 |
| 30 | 0.233 | 238.0 | 226.5 | 166.4 | 0.701 | 0.736 | 0.907 |
| 40 | 0.233 | 236.0 | 223.9 | 169.2 | 0.719 | 0.757 | 0.899 |
| 50 | 0.233 | 234.8 | 222.3 | 171.1 | 0.731 | 0.771 | 0.895 |

### Inspection criterion (severe scenario)

* allowable life at gamma = 1.50, p05 capacity: 14.96 years (governed).
* allowable mean pit depth at 25 years: 5.24 mm (0.349 t) = 0.210 mm/yr, against the assumed 0.289 mm/yr (margin factor 0.72).
* baseline scenario: allowable life None (beyond_horizon); allowable mean pit depth 10.22 mm (margin factor 6.49) -- 25 years of Melchers wastage (1.57 mm mean pit depth) is far from governing.

| years | sigma_u p05 [MPa] | gamma x sigma_ice p95 [MPa] | margin | U p50 | next inspection [yr] | action |
|---|---|---|---|---|---|---|
| 0 | 262.4 | 216.5 | 1.212 | 0.550 | 5.0 | routine |
| 5 | 258.1 | 221.1 | 1.167 | 0.567 | 5.0 | routine |
| 10 | 247.8 | 228.7 | 1.083 | 0.599 | 5.0 | increase_frequency |
| 15 | 237.8 | 238.0 | 0.999 | 0.642 | 0.0 | renew_or_dock |
| 20 | 231.7 | 244.6 | 0.947 | 0.671 | 0.0 | renew_or_dock |
| 25 | 228.4 | 249.1 | 0.917 | 0.688 | 0.0 | renew_or_dock |
| 30 | 226.5 | 252.2 | 0.898 | 0.701 | 0.0 | renew_or_dock |
| 40 | 223.9 | 256.8 | 0.872 | 0.719 | 0.0 | renew_or_dock |
| 50 | 222.3 | 260.2 | 0.854 | 0.731 | 0.0 | renew_or_dock |

Action vocabulary: `routine` (next survey within the interval cap), `increase_frequency` (margin below 1.1), `renew_or_dock` (allowable life reached, renewal or docking required), `close_out` (target unreachable).