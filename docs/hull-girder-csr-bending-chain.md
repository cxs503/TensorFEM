# Hull-girder longitudinal bending chain: wave load → beam FE → CSR/DNV limit state

Mission IN50-N07 (campaign node, physics domain). Repository: **TensorFEM**
(`tensorfem.hull_girder_csr_chain`). Benchmark ship: **KVLCC2**.

This document is the evidence trail for the module: what is computed, with which
formulas, which numbers were produced, what the numbers are compared against,
and what the model does **not** cover.

---

## 1. Case introduction

### 1.1 Physical background

A ship hull girder is a box beam loaded transversely by the difference between
buoyancy and weight. Two load systems matter for longitudinal strength:

* **still water**: the actual weight distribution of the loaded ship against the
  hydrostatic buoyancy of the trimmed waterline;
* **waves**: the change of immersed volume when a wave passes the hull, which
  cannot change displacement or trim, so the wave adds a self-equilibrated load
  system along the length.

Their sum produces the sagging (deck in compression, midship deflected
downwards) and hogging (deck in tension) bending moments that the midship
section must carry. Classification rules (DNV GL 2019 / IACS UR S11) prescribe
the minimum wave bending moment and the minimum deck section modulus; CSR
prescribes permissible hull-girder stress. Beyond the elastic check the section
must survive up to its ultimate (collapse) moment.

### 1.2 Geometry and parameters (KVLCC2, public particulars)

| Quantity | Symbol | Value |
|---|---|---|
| Length between perpendiculars | L | 320.0 m |
| Moulded breadth | B | 58.0 m |
| Moulded depth | D | 30.0 m |
| Design draft | T | 20.8 m |
| Block coefficient | Cb | 0.8095 |
| Displacement (ρ·Cb·L·B·T) | Δ | 320 319 t (design 320 000 t, +0.1 %) |
| L/B, B/T, B/D | — | 5.52, 2.79, 1.93 |
| Double-bottom depth | — | 2.7 m |
| Longitudinal bulkheads | — | 2, from 2.7 m to 30 m |
| Steel | — | E = 206 GPa, σ_y = 235 MPa (mild steel) |
| Material factor | γ_m | 1.43 (mission spec) |
| CSR permissible hull-girder stress | — | 220 MPa |
| Design wave height (100-yr North Atlantic, L > 300 m) | h_w | 13.5 m |
| Water | — | ρ = 1025 kg/m³, g = 9.81 m/s² |

The geometry is verified **before** any load is applied:
`HullForm.verify()` integrates the sectional-area curve and reports
`Cb_error = 1.4e-16`, `displacement_t = 320 318.5` and `LCB = 0.500 L`
(symmetric idealisation). The chain raises `RuntimeError` if `Cb_error > 1e-3`.

### 1.3 Midship scantling set

The KVLCC2 **hull form** is public; the **structural drawings are not**. The
midship section used here is therefore an explicit CSR-consistent
reconstruction (`kvlcc2_midship_section`), documented member by member:

| Member | Area (m²) | Height y (m) | Plate t (mm) |
|---|---|---|---|
| deck plating | 1.74 | 29.985 | 30.0 |
| deck longitudinals (0.9 m spacing) | 0.773 | 29.80 | 30.0 (attached plating) |
| bottom shell | 1.856 | 0.016 | 32.0 |
| bottom longitudinals | 0.773 | 0.30 | 32.0 |
| inner bottom | 1.288 | 2.714 | 28.0 |
| inner-bottom longitudinals | 0.511 | 2.55 | 28.0 |
| hopper plating (2×) | 0.264 | 1.35 | 24.0 |
| side shell (2×, 27.3 m high) | 1.420 | 16.35 | 26.0 |
| side longitudinals (2×) | 0.607 | 16.35 | 26.0 |
| longitudinal bulkheads (2×) | 1.310 | 16.35 | 24.0 |

Total area 10.60 m². The set is closed against the rule: IACS UR S11
`W_min = C L² B (Cb + 0.7) = 96.38 m³` (`C = 10.75` for 300 m ≤ L ≤ 350 m) and
the deck plating is thickened from 30.0 mm to **30.97 mm** by bisection until
`W_deck ≥ W_min`. Resulting properties:

| Quantity | Value |
|---|---|
| neutral axis above keel | 12.914 m |
| I_midship | **1646.7 m⁴** |
| W_deck (y = 17.086 m) | **96.38 m³** (= UR S11 minimum) |
| W_bottom (y = 12.914 m) | **127.51 m³** |
| centroid residual `|∫y dA| / ∫|y| dA` | 0.0 (exact) |

> **Correction of a previous claim.** An earlier module in this repository
> (`hull_girder_strength.py`) quoted `I_midship ≈ 1.15e7 m⁴` as a "Kim et al.
> 2018" reference and used component areas of order 10⁴ m². A 320 m tanker deck
> modulus of ~96 m³ corresponds to `I ≈ 1.6e3 m⁴`; 1.15e7 m⁴ is three orders of
> magnitude too large (it would give deck stresses of ~0.1 MPa). That reference
> value is withdrawn and is not used anywhere in the new chain.

---

## 2. Method

### 2.1 Chain overview

```
hull form A(x) ──► buoyancy ρg A(x) ──┐
                                      ├─► still-water load q_SW(x) ─┐
block weights w(x) ───────────────────┘                             │
                                                                    ├─► beam FE (EI(x)) ─► M(x) ─► σ(x,y) ─► CSR & ultimate
design wave h_w, λ=L ─► δA = B_wl η e^{-kT} ─► q_W(x) ─────────────┘
```

1. **Hull form** – symmetric sectional-area curve `A(ξ) = A_mid sin(πξ)^0.5`
   scaled numerically so `∫A dx = Cb L B T`; wall-sided sections so
   `B_wl = A/T`.
2. **Weights** – block masses (hull steel 0.20Δ ∝ A(x), cargo 0.70Δ uniform over
   0.18–0.92 L, machinery 0.035Δ aft, outfitting 0.025Δ, remaining systems
   0.04Δ), each normalised on the discrete stations so its trapezoidal integral
   equals its mass exactly. LCG = 163.24 m from AP (0.510 L).
3. **Equilibrium** – draft scale and longitudinal shift solved (damped Newton,
   2 unknowns) so buoyancy balances mass *and* moment:
   force residual 6.5e-15 relative, moment residual 3.9e-14 relative.
   Draft 20.84 m, trim 3.52 m (LCG aft of the symmetric buoyancy centre).
4. **Wave load** – regular design wave `η = (h_w/2) cos(2π(x−x_c)/λ)`, crest
   amidships (hogging), `λ = L`, Smith correction `e^{-kT}` (T = 20.8 m,
   `kT = 0.408`). Immersed-area change `δA = B_wl η e^{-kT}` is de-meaned and
   de-trimmed per station (a wave cannot change displacement or trim), leaving
   `q_W` with zero net force and zero net moment
   (checked: `|∫q_W|/max|q_W| < 1e-9`).
5. **Equivalent beam** – 320 two-node Hermite Euler–Bernoulli elements with
   `EI(x) = E I_mid (A/A_max)(0.55 + 0.45 A/A_max)²` (2 % floor at the extreme
   stations). Consistent nodal loads by 2-point Gauss quadrature, pinned ends.
   Bending moment recovered as `M = EI v''` at the nodes; stress as
   `σ = −E v'' y = −M y / I`.
6. **Rule comparison** – DNV GL 2019 / IACS UR S11:
   `M_wh = 0.19 f_nl f_m f_h C L² B Cb` and
   `M_ws = 0.11 f_nl f_m f_h C L² B (Cb + 0.7)` [kN·m], `f_nl = f_m = f_h = 1`.
   Total design moments composed as `|M_SW − M_wh|` (hogging) and
   `|M_SW + M_ws|` (sagging), with `M_SW` the computed still-water moment
   (sagging positive).
7. **CSR check** – `σ_allow = min(σ_y/γ_m, 220 MPa)`,
   `SF = σ_allow/σ`, `utilisation = M_total/M_u`.
8. **Ultimate strength** – Smith-type incremental-curvature collapse on the same
   section (`tensorfem.hull_girder_progressive.solve_component_section`, plane
   sections + axial equilibrium). Tension: elastic–perfectly plastic at σ_y.
   Compression: `min(σ_y, JO(σ_E,plate), JO(σ_E,column))` with
   `σ_E,plate = 4π²E/(12(1−ν²))(t/b)²` and `σ_E,column = π²E r²/s²`
   (frame spacing `s = 5.0 m`, `r = 0.055 m` for a 400 mm bulb flat with
   effective plating). Implementing the Johnson–Ostenfeld transition gives
   σ_c = 178.9 MPa (0.76 σ_y) for the deck/bottom plating — the source of the
   sagging/ hogging asymmetry.

### 2.2 Tool chain and reproducibility

| Item | Value |
|---|---|
| module | `src/tensorfem/hull_girder_csr_chain.py` |
| tests | `tests/test_hull_girder_csr_chain.py` (23 tests, one group per criterion) |
| runner | `examples/hull_girder_csr_chain.py` |
| evidence bundle | `artifacts/hull_girder_csr_chain/{results.json,*.png}` |
| command | `PYTHONPATH=src python examples/hull_girder_csr_chain.py` |
| test command | `PYTHONPATH=src python -m pytest tests/test_hull_girder_csr_chain.py -q` |
| dependencies | numpy (numerics), matplotlib (figures only, lazy import) |

---

## 3. Results

### 3.1 Wave and still-water moment (midship)

| Quantity | Value |
|---|---|
| DNV/UR S11 `M_wh` | 9.820 GN·m |
| DNV/UR S11 `M_ws` | 10.601 GN·m |
| chain `M_W` (h_w = 13.5 m, λ = L, Smith) | −10.121 GN·m (hogging) |
| rule-equivalent wave height implied by `M_wh` | 13.099 m |
| chain / rule wave moment ratio | 1.031 |
| chain `M_SW` (sagging positive) | +0.534 GN·m |
| `M_total,hog` (chain) | **9.587 GN·m** |
| `M_total,sag` (chain) | **10.654 GN·m** |
| `M_total,hog` (rule composed, `M_SW − M_wh`) | 9.286 GN·m |
| `M_total,sag` (rule composed, `M_SW + M_ws`) | 11.135 GN·m |

The physical 100-yr design wave with Smith correction reproduces the rule wave
moment to **3.1 %** *before* any calibration — the Holtrop-form coefficient that
matches the rule at `h_w = 13.5 m` is `k_w,hog = 1.505`, `k_w,sag = 1.624`
(dimensionless), i.e. two orders of magnitude away from the `k_w = 11.6`
recorded in the earlier draft.

### 3.2 Elastic stresses

| Case | σ_deck (MPa) | σ_bottom (MPa) |
|---|---|---|
| hogging | **+99.5** (tension) | −75.2 (compression) |
| sagging | −110.6 (compression) | **+83.6** (tension) |

Allowable stress `σ_allow = min(235/1.43, 220) = 164.3 MPa` (γ_m governs).

### 3.3 CSR / DNV limit state

| Case | M (GN·m) | W (m³) | σ (MPa) | σ_allow (MPa) | SF | M_u (GN·m) | M/M_u |
|---|---|---|---|---|---|---|---|
| hogging (deck) | 9.587 | 96.38 | 99.5 | 164.3 | **1.652** | 25.34 | **0.378** |
| sagging (bottom) | 10.654 | 127.51 | 83.6 | 164.3 | **1.967** | 22.20 | **0.480** |

Ultimate capacity: hogging 25.34 GN·m (shape factor 1.118 over the first-yield
moment of 22.67 GN·m), sagging 22.20 GN·m (1.286 over 17.26 GN·m — the sagging
first-yield moment is deck-compression governed at 178.9 MPa).

### 3.4 Acceptance criteria (machine-checked)

| Criterion | Definition | Result | Verdict |
|---|---|---|---|
| C1 | `|I_paraxial − I_strip| / I_strip` | 3.1e-6 (I = 1646.7 m⁴) | **PASS** (≤ 5 %) |
| C2 | `|M_total,hog − (M_SW − M_wh)| / (…)` | 3.24 % (sagging 4.32 %) | **PASS** (≤ 10 %) |
| C3 | stress / moment vs analytic `M y/I` and `∫∫q` | 0.202 % | **PASS** (≤ 1 %) |
| C4 | `SF = σ_allow/σ` reported | 1.652 / 1.967 | **PASS** (≥ 1) |
| C5 | `M_total/M_ultimate` | 0.378 / 0.480 | **PASS** (≤ 1) |
| C6 | pytest coverage of C1–C5 | 23 tests pass | **PASS** |
| C7 | figures | 4 PNG (moment, stress, section, M–κ) | **PASS** |

### 3.5 Figures

| Figure | Content |
|---|---|
| `docs/assets/hull-girder-csr/moment_distribution.png` | still-water and wave load, plus `M_SW`, `M_W`, total hogging/sagging moments along the hull with the analytic double integration overlaid |
| `docs/assets/hull-girder-csr/stress_distribution.png` | deck and bottom longitudinal stress along the hull for both cases with the allowable stress line |
| `docs/assets/hull-girder-csr/section_stress.png` | linear stress distribution over the midship depth (neutral axis marked) |
| `docs/assets/hull-girder-csr/ultimate_moment_curvature.png` | moment–curvature collapse curves with the design moments marked |

---

## 4. Pitfalls and limitations

1. **Units/coefficient trap (fixed).** `M_w = k_w ρ g B L² Cb h_w / 100` with
   `k_w = 11.6` yields 75.8 GN·m for KVLCC2 — a factor 7.7 above the rule value.
   The coefficient implied by DNV/UR S11 is 1.505 (hogging) at `h_w = 13.5 m`,
   and `equivalent_wave_coefficient` now derives it instead of hard-coding it.
   A regression test pins `k_w = 11.6 ⇒ M_w / M_rule > 5`.
2. **Fictitious reference inertia (fixed).** See §1.3. C1 is therefore evaluated
   as a *dual-algorithm* consistency check (parallel-axis vs 200-strip
   integration) plus a *rule anchor* (`W_deck ≥ W_min` from IACS UR S11). No
   direct literature `I_midship` for KVLCC2 is used, because the public KRISO
   data set describes the hull **form**, not the scantlings; quoting a number
   without a verifiable scantling set would be fabrication.
3. **Scantling set is a reconstruction.** `W_deck` is *closed onto* the UR S11
   minimum, i.e. the section is a rule-minimum section by construction. Real
   ships usually exceed the minimum by 5–20 %, so the resulting stresses are
   conservative (high) but the section modulus is not a survey of a built ship.
4. **Wall-sided, symmetric hull.** No flare, no bulbous bow, no rise of floor in
   the hydrostatics; the Smith correction uses the draft only (`e^{-kT}`), not a
   section-shape factor. `LCB = 0.5 L` is imposed by symmetry; the real KVLCC2
   hull has a small LCB offset, which would shift `M_SW` by a few 10⁸ N·m.
5. **Beam model is linear elastic.** No shear lag (deck stress near the
   hatch/side intersection is under-predicted), no warping/torsion coupling, no
   dynamic amplification (`f_nl` style factors are not applied).
6. **`EI(x)` is an idealisation** with a 2 % floor at the extreme stations; the
   midship moment is insensitive to it (max moment deviation between FE and
   closed-form integration is 0.2 %, dominated by load interpolation on the
   beam grid).
7. **Ultimate strength excludes** stiffener tripping, tripping/plate
   interaction, residual stress, initial imperfections, corrosion, load
   shedding along the length, and cyclic/unloading effects. The deck compression
   limit (178.9 MPa = 0.76 σ_y) is a Johnson–Ostenfeld estimate; with tripping
   included `M_u` would drop by roughly 5–15 %. The reported `M/M_u ≈ 0.4–0.5`
   has ample margin against that uncertainty, but the absolute `M_u` should not
   be read as a class-rule ultimate strength assessment.
8. **The DNV factors are assumed unity** (`f_nl = f_m = f_h = 1`). They are
   explicit function arguments, so a rule assessment with other factors is a
   one-line change, but the reported rule moments are for that assumption.

---

## 5. Error verdict (explicit)

| Key quantity | Reference | Error | < 3 %? |
|---|---|---|---|
| `I_midship = 1646.7 m⁴` | 200-strip integration of the same members (1646.67 m⁴) | 0.0003 % | **yes** |
| `W_deck = 96.38 m³` | IACS UR S11 `W_min = 96.38 m³` | 0.00 % (closure target) | **yes** (by construction, see §4.2) |
| `M_total,hog = 9.587 GN·m` | DNV/UR S11 composed 9.286 GN·m | **3.24 %** | no (3.2 % > 3 %) — inside the 10 % criterion |
| `M_total,sag = 10.654 GN·m` | DNV/UR S11 composed 11.135 GN·m | **4.32 %** | no — inside the 10 % criterion |
| `M_W` (chain) | DNV/UR S11 `M_wh` | 3.06 % | no (3.1 % > 3 %) — inside the 10 % criterion |
| `σ_deck,hog = 99.47 MPa` | analytic `−M y/I` | 0.000 % (identical formula) | **yes** |
| `M(x)`, whole length | closed-form double integration | 0.202 % | **yes** |
| `M_u,hog = 25.34 GN·m` | no independent experimental/benchmark value | n/a (self-consistent Smith model) | **no reference — declared** |
| `M_u,sag = 22.20 GN·m` | as above | n/a | **no reference — declared** |

Honest statement: C2 (the only criterion with an external, non-self-referential
reference) sits at 3.2–4.3 %, i.e. **above** the 3 % "publication-grade" line but
**inside** the 10 % acceptance band of the mission. The reason is physical, not
numerical: the chain uses a real design wave (13.5 m, its own Smith-corrected
hydrostatics) while the rule moment is an empirical envelope; a 3 % difference
is expected. C1 is exact by construction of the closure loop; the *independent*
part of C1 is the 0.0003 % agreement between two different integration
algorithms. The ultimate moments have **no** trustworthy published reference for
this reconstructed section, and no such comparison is claimed.

---

## 6. What was not done

* No comparison against the published KVLCC2 ISSC/CSR ultimate-strength
  benchmark section (scantlings not verifiable from the public data set).
* No shear-lag / grillage / FE-shell cross-check of the deck stress at the
  deck–side-shell connection.
* No dynamic (springing/whipping) amplification and no fatigue assessment.
* No corrosion allowance or net/gross scantling distinction.
* No interface work on the ship-performance-platform side (owned by that
  repository); this delivery is TensorFEM-only, as required by the charter.