# Ice-induced slamming and hull-girder whipping: load, transient and fatigue increment

Mission `m9892`, campaign node `p9050-n04-ice-slamming-whipping`.
Module: `tensorfem.ice_slamming_whipping` (new). Artifact:
`benchmarks/ice_slamming_whipping/result.json`, produced by
`benchmarks/ice_slamming_whipping/make_result.py`. Tests:
`tests/test_ice_slamming_whipping.py` (26 tests, all passing).

Status: **4 of the 5 machine criteria pass; criterion (a) is an honest
WITHHELD/FAIL** because no authorised ice model-test force table is reachable
from this environment (§6). No number anywhere in this document or in the
artifact was transcribed from memory: every value in the tables is computed by
the script named above, and the artifact records the script and command.

---

## 1. Case introduction

An ice-class bulker is transiting a brash-ice field in a seaway. Two load
mechanisms act on the forebody and both are impulsive compared with the hull
girder bending period (about 2 s):

1. **ice impacts** — the forebody rams individual floes at transit speed;
   crushing pressure on the ice-contact area `A = h·b_c` over a few tens of
   milliseconds;
2. **water-entry slamming** — when the relative bow motion exceeds the local
   immersion the forebody leaves the water and re-enters with vertical velocity
   `w`; the load is the rate of change of section added mass (von Karman wedge
   entry).

Either pulse excites the global hull-girder bending modes, i.e. **whipping**,
and whipping is a fatigue driver because it adds stress cycles at a frequency
(≈1-2 Hz) that is fast compared with the wave cycle and outside the range of the
wave bending moment.

| quantity | symbol | value | grade |
|---|---|---|---|
| length between perpendiculars | L | 230.0 m | CASE_INPUT (declared) |
| breadth | B | 32.0 m | CASE_INPUT |
| mean draft | T | 13.5 m | CASE_INPUT |
| block coefficient | C_b | 0.80 | CASE_INPUT |
| displaced volume / mass | ∇ / Δ | 79 488 m³ / 81 475 t | computed from L,B,T,C_b |
| L/B, B/T, L/T | — | 7.19, 2.37, 17.04 | computed, inside the usual bands |
| moulded depth (girder depth) | D | 20.0 m | CASE_INPUT |
| equivalent wall thickness | t | 0.040 m | CASE_INPUT |
| girder inertia / section modulus | I / Z | 307.1 m⁴ / 30.77 m³ | computed |
| impact station (forward of midships) | x_s | 103.5 m | CASE_INPUT |
| local immersion at that station | T_s | 3.5 m | CASE_INPUT |
| waterline half breadth there | b_c | 6.5 m | CASE_INPUT |
| wetted forebody length | L_s | 14.0 m | CASE_INPUT |
| forebody deadrise | β | 30° | CASE_INPUT |
| ice thickness / concentration | h / C | 0.5 m / 0.6 | CASE_INPUT (gate case) |
| floe length (mean, lognormal) | — | 12.0 m, cv 0.35 | DECLARED |
| ice crushing pressure (reference) | p_ref | 0.66 MPa at 1 m², 1 m/s | DECLARED, swept 0.40-1.00 MPa |
| transit sea state | H_s / T_p | 2.0 m / 7.0 s | CASE_INPUT |
| slam sea state (emergence case) | H_s / T_p | 6.0 m / 9.5 s | CASE_INPUT |
| speed | V | 10 kn = 5.144 m/s | gate case, swept 4-13 kn |
| vibrating mass per unit length | m_struct | 354 240 kg/m | Δ/L |
| hydrodynamic added mass per unit length | m_added | 644 351 kg/m | 0.70 × Lewis 2D heave |

Geometry is validated before any physics: :meth:`IceClassShipCase.geometry_checks`
recomputes the displaced volume from C_b·L·B·T, reproduces the mass from the
water density and flags the bulk ratios outside the standard bands. All three
flags are false for the declared hull.

## 2. Method

### 2.1 Hull-girder model (free-free, Timoshenko)

The girder is a uniform equivalent box girder, discretised with the standard
shear-flexible beam element (dof order `(w, ψ)`), stiffness parameter
`φ = 12EI/(κGA L_e²)`, consistent translational mass plus the steel rotary
inertia `ρ_I·I`. Boundary conditions are **free-free**: a floating hull has no
supports, and pinning it would raise the fundamental bending frequency by a
factor 2.27 (the resulting model would be neither the ship nor the textbook
beam). The generalised eigenproblem is solved in mass-normalised form after a
Cholesky factorisation of `M`; the two rigid-body roots (heave, pitch) are
counted and skipped. Bending moment is recovered from the eigenvector rotation
field, `M(x) = EI·dψ/dx`, evaluated at element midpoints.

### 2.2 Motion model

2-DOF linear heave/pitch in head seas: mass `Δ` and `Δ k_yy²`, added mass from
the qualified Lewis 2D heave coefficient of `tensorfem.hull_girder_modal`
multiplied by the length (pitch by parallel-axis scaling), restoring
`ρg A_wp` and `ρg ∇ GM_L`, radiative damping as declared ratios of the wet
natural frequencies. The Froude-Krylov force and moment are the exact integrals
of `ρgζ_a·B·e^{ikx}` over `[-L/2, L/2]`. The relative motion at the impact
station is the sum of three phasors (heave, pitch, wave elevation) — the
re-entry velocity follows in closed form from the harmonic amplitude:
`w = ω_e·sqrt(a² − T_s²)` when `a > T_s`, zero otherwise.

### 2.3 Loads

*Ice impact.* Inter-arrival spacing `s/CV` from the transit rate; floe lengths
from a seeded lognormal (the train is exactly reproducible). Contact area
`A = h·min(2b_c, l_floe)`, peak `F = p(A,V)·A` with a Sanderson-type
pressure-area softening and a declared strain-rate hardening term,
`p = p_ref (A/A_ref)^{-0.30} (V/V_ref)^{0.25}`. Pulse duration matches the floe
momentum `m_floe·V` with a triangular pulse — that is why the pulse is only
20-30 ms long, and why the load behaves as an impulse for a 2 s bending mode.

*Water-entry slam.* von Karman wedge entry: peak `F = πρ b_c L_s w² cot β`,
rise time `t_rise = b_c tan β / w` (time to wet the chine), triangular pulse.
Wagner's spray-root width `(π/2)c` gives the upper bound of the formulation
(×2.467 in force); the reported value is von Karman's.

### 2.4 Whipping and fatigue

Each modal coordinate is integrated **exactly** for a piecewise-linear load: the
state transition of the damped oscillator is closed form, so the response is
exact to machine precision and the impulse of a 20 ms pulse is not lost to
sampling (the time grid is the uniform response grid *merged with every pulse
vertex*). The midship bending moment is `M = EI·Σ q_i ψ_i'`, the hot-spot stress
is `SCF·M/Z` (SCF = 1.15), and fatigue is rainflow counting (reusing
`tensorfem.marine_fatigue_advanced.rainflow_cycles`) plus a Miner sum on the
DNV-RP-C203 curve D in air (`N = 1.52e12 S^{-3}`, knee 53.4 MPa, `m = 5` above
the knee; coefficients are published in MPa, the module converts internally and
keeps an SI interface).

The three damage quantities are kept separate and are defined once here:

* `D_wave` — damage of the **wave-induced vertical bending moment** history
  (self-equilibrated linear strip load on the box hull: the wave-excess buoyancy
  minus the rigid-body heave/pitch reaction, no Smith correction, no
  diffraction). This is the wave-fatigue baseline.
* `ΔD_slam` — damage of the **ice-impact** bending moment history. This is the
  quantity of the campaign node.
* `D_combined` — damage of `σ_wave + σ_slam` rainflow-counted together, i.e. the
  interaction of the two stress processes (shows the coupling of §5).

## 3. Results

### 3.1 Gate case: V = 10 kn, h = 0.5 m, C = 0.6, H_s = 2 m

| quantity | value |
|---|---|
| ice-impact pulses in 120 s | 31 (spacing 3.89 s) |
| peak ice-impact force | **3.685 MN** |
| peak water-entry slam force | 0 MN (no bow emergence in a 2 m sea) |
| peak whipping bending moment (midships) | **123.2 MN m** |
| quasi-static moment of the same load | 20.9 MN m → DAF = 5.89 |
| wave-induced VBM (baseline) | 103.2 MN m |
| modal frequencies f₂, f₃, f₄ | **0.506, 1.225, 2.060 Hz** |
| modal share of the whipping moment | 62.2 %, 17.4 %, 13.1 % (2-, 3-, 4-node) |
| whipping decay time τ | **12.59 s** (analytic `1/(ζω)` = 12.58 s) |
| `D_wave` (120 s) | 1.573e-10 |
| `ΔD_slam` (120 s) | 1.248e-10 |
| `ΔD_slam / D_wave` | **0.79** |
| `D_combined / D_wave` | **9.71** |

The dynamic amplification factor is measured against the *quasi-static moment of
the same force*, not against the classical SDOF formula; for an impulse much
shorter than the mode period it is legitimately large, so it is reported with
its reference moment.

### 3.2 Frequency evolution and the (V, h) grid

Figures: `docs/assets/ice-slamming-whipping/`.

| figure | content |
|---|---|
| `load-and-whipping.png` | ice-impact pulses, the whipping `M_v(t)` they produce, and the wave VBM with/without whipping |
| `frequencies.png` | f₂-f₄: FE vs exact Timoshenko vs Euler-Bernoulli |
| `decay.png` | isolated whipping transient and the fitted 1/e envelope |
| `force-and-damage-trends.png` | peak force vs h for three speeds against the quoted literature band; `ΔD_slam/D_wave` vs speed |

| V [kn] | h [m] | peak force [MN] | peak whipping M [MN m] | ΔD_slam / D_wave (60 s) |
|---|---|---|---|---|
| 10.0 | 0.3 | 2.577 | 65.2 | 0.040 |
| 7.0 | 0.5 | 3.371 | 32.6 | 0.0011 |
| 10.0 | 0.5 | 3.685 | 106.6 | 0.485 |
| 10.0 | 0.8 | 5.120 | 166.9 | 4.74 |

All four rows are computed by the artifact; the damage column is per 60 s
exposure, which is why it differs from the 120 s gate case of §3.1.

The jump of the damage ratio by two orders of magnitude between h = 0.3 m and
h = 0.8 m is the `S^{-3}` slope multiplied by the resonant build-up: at
V = 10 kn the inter-arrival time is 3.89 s against a whipping period of 1.98 s,
i.e. the impact train sits close to a 2:1 ratio with the 2-node mode, so
successive transients *superpose* instead of decaying (τ = 12.6 s ≫ 3.89 s).
This is a real mechanism and it is reported as such, but the single deterministic
floe train does not resolve its statistics: a converged answer needs a Monte
Carlo over floe sizes and arrival phases (declared limit).

## 4. The coupling of ice and wave slamming

In the gate case the sea state does not emerge the bow, so the wave-slam branch
is identically zero and the coupling question is empty. It is therefore
evaluated in the slam sea state (H_s = 6 m, T_p = 9.5 s), where the relative bow
motion (3.57 m) just exceeds the 3.5 m immersion:

| quantity | value |
|---|---|
| re-entry velocity | 0.637 m/s (marginal emergence) |
| von Karman peak slam force | 0.206 MN (Wagner upper bound 0.509 MN) |
| rise-phase impulse vs added-mass momentum `m_a·w` | 606902 N s vs 606902 N s, residual 0 |
| peak ice force / peak total force | 3.685 / 3.871 MN |
| equivalent Ochi coefficient `F/(0.5 rho A_s w^2)` | 5.44 |
| `D_wave` / `ΔD_slam`(ice) / combined | 6.682e-06 / 3.818e-11 / 6.876e-06 |
| `damage_superposition_ratio` | 1.029 |

The qualitative result is:

* the two branches superpose in the *stress* domain, not in the *damage* domain;
  the rainflow count of the summed stress gives more damage than the sum of the
  separately counted damages (`damage_superposition_ratio` > 1), because ranges
  add before the `S^{-3}` weighting;
* the equivalent Ochi coefficient inferred from the water-entry branch (5.44) is
  far outside the 0.6-1.0 empirical band because the emergence here is marginal
  (`w = 0.64 m/s`): the `0.5ρA_s w²` normalisation is not the right one for a
  chine-limited wedge entry, which is why the module reports the von Karman
  formulation and the equivalent coefficient separately;
* the water-entry pulse is far longer (≈0.8 s) than the ice pulse (≈0.02 s), so
  it excites the 2-node mode quasi-statically while the ice pulse is a pure
  impulse — the modal shares of the two responses differ;
* phase matters only through the peak of the sum; for the fatigue the
  superposition of the *processes* is what counts.

## 5. Verification: what each number is checked against

| check | reference | measured |
|---|---|---|
| wet girder f₂-f₄ | exact free-free Timoshenko dynamic-stiffness solution (`tensorfem.dsbeam.natural_frequencies_hz`), same girder | **max 1.11 %** (0.31 / 0.66 / 1.11 %) |
| free-free Euler-Bernoulli roots | `cos(x)cosh(x) = 1` evaluated at the computed root | residual < 1e-12 |
| bending-moment recovery | static `P·a·b/L` of the same FE girder, self-equilibrated load | **5.5e-12** relative |
| modal integrator | independent RK4 integration of the same SDOF ODE | < 1e-6 relative (test) |
| decay time | analytic `1/(ζω)` | **0.06 %** |
| wave-slam peak | von Karman added-mass momentum `m_a·w` = rise-phase impulse | **< 1e-9** relative (test) |
| ice crushing law | `∂lnF/∂lnh = 1+α`, `∂lnF/∂lnV = γ` of the declared law | measured 0.70 / 0.25 vs expected 0.70 / 0.25 |
| negative controls (V→0, h→0, C→0) | limits, not coded branches | **0.0, < 1e-9, 0.0** |

Shear flexibility is *not* a numerical artefact: the free-free Euler-Bernoulli
values are 0.536 / 1.478 / 2.898 Hz and the Timoshenko values 0.506 / 1.225 /
2.060 Hz, i.e. shear softening of 5.4 % / 16.6 % / 28.1 % for a thin-walled box
girder (κGA is small because the equivalent wall is 40 mm). Criterion (b) is
therefore judged against beam **theory including shear**, and the
Euler-Bernoulli deviation is reported as a physical contribution, not as error.

## 6. Criteria: verdicts

| id | criterion | verdict | measured |
|---|---|---|---|
| (a) | peak slam force vs ice model-test literature, ≥ 3 (V,h) combinations, error < 20 % | **WITHHELD (honest FAIL)** | no authorised table obtainable; attempts logged in the artifact |
| (b) | f₂-f₄ vs hull-girder beam theory, error < 5 % | **PASS** | 1.11 % max |
| (c) | τ_decay inside the 5-15 s literature window, error < 30 % | **PASS** | 12.59 s; measured vs analytic 0.06 % |
| (d) | V→0, h→0, C→0 hard negative controls | **PASS** | residuals 0.0 / <1e-9 / 0.0 |
| (e) | F_slam_max(V=10 kn, h=0.5 m) must not fall below 1 MN | **PASS** | 3.685 MN, inside the quoted 2-5 MN band |

On (a): the *only* honest statement available is a **range-overlap consistency
argument**. The declared crushing-pressure level has a band from 0.40 MPa to
1.00 MPa, which moves the gate-case peak force from 2.23 MN to 5.58 MN; that band
overlaps the 2-5 MN quoted for model tests. Overlap is not agreement and is
labelled as such in `result.json → criteria[0].substitute_evidence`.

On (c): the decay time is a function of the declared structural damping,
`τ = 1/(ζω)`. The 5-15 s window is equivalent to ζ ∈ [2.10 %, 6.29 %] at
f₂ = 0.506 Hz; the declared ζ = 2.5 % sits inside that band. This is a
consistency check between the window and the declared damping, not an
independent validation of either.

## 7. Points to note, limits and applicability

* **Free-free, not pinned.** This was the single largest modelling decision: a
  pinned girder halves the frequency (§2.1). The whipping frequencies are
  consistent with the campaign node's guidance (f₂ 0.51 Hz in 0.5-1.0; f₃ 1.22 Hz
  in 1.0-1.5; f₄ 2.06 Hz marginally above the 1.5-2.0 guidance) even though the
  uniform girder follows the `n²` law (f₃/f₂ = 2.42) rather than the 1.8-2.2 of a
  real non-uniform hull with concentrated masses.
* **The wave-slam branch is the weakest link of the motion model.** The
  Froude-Krylov-only 2-DOF model cancels over a hull longer than one wavelength
  (kL/2 = 9.5 rad at T_p = 7 s), so the pitch response is an order of magnitude
  below measured pitch RAOs and the relative bow motion is dominated by the wave
  elevation itself. The wave branch is therefore only credible as a *threshold*
  model: it fires only when emergence is large, and the slam magnitude inherits
  the relative-velocity error.
* **The ice crushing law is declared, not measured.** `p_ref`, the two
  exponents and the floe-size distribution set the force level; the sensitivity
  band is 2.2-5.6 MN. This is the dominant uncertainty of §3 and the reason
  criterion (a) cannot be closed.
* **Rotation of the water.** The added mass is translational; the rotary inertia
  is that of the steel only. Adding the water's rotary inertia would lower the
  higher modes further.
* **Deterministic impact train.** One seed, no Monte Carlo; the resonant build-up
  of §3.2 is real but its *statistics* are not resolved.
* **Damage is reported per exposure** (120 s gate case, 60 s grid) because it is
  linear in exposure only in the small-damage limit and the ratio is not exactly
  scale invariant (the rainflow count of a pulse train with a ratio near a
  sub-harmonic of the whipping period is sensitive to the window length).
* **V→0 control.** The ice branch and the ice-induced whipping collapse exactly.
  The *water-entry* branch does not: a ship stopped in a storm still slams, and
  its whipping decay time stays finite (12.65 s in the slam sea state). That is
  reported next to the ice-only number rather than hidden in the definition; the
  criterion is evaluated on the ice branch, which is what the node asks about.

## 8. Error judgement against references, per quantity

| quantity | reference available | error | < 3 %? |
|---|---|---|---|
| f₂, f₃, f₄ | exact Timoshenko DSM (same girder) | 0.31 / 0.66 / 1.11 % | yes |
| midship moment of a static load | `P·a·b/L` | 5.5e-12 | yes |
| modal response | independent RK4 | < 1e-6 (test) | yes |
| τ_decay | analytic `1/(ζω)` | 0.06 % | yes |
| slam peak | von Karman momentum closure | < 1e-9 (test) | yes |
| **peak slam force** | **ice model-test table** | **no reference obtained** | **cannot be judged → WITHHELD** |
| damage ratio ΔD_slam/D_wave | none (no benchmark exists) | — | no basis; reported as a model result with its declared input band |

The one quantity the campaign node most cares about (the force level against
measurements) is exactly the one that cannot be closed here. That is stated
rather than patched with a citation that does not exist.