# Stiffened-panel ultimate strength with ISO 19906 / CSR screening

Status: **first evidence package**, admitted only for the quantities listed in
*Acceptance results*. Two literature comparisons are explicitly **WITHHELD**
(Smith 1977 test series, ABAQUS Riks reference solution) because neither source
is available in this environment; nothing in this document quotes a value from
memory.

Modules:

* `tensorfem.stiffened_panel_ultimate` — component limiting strengths,
  load-end shortening, Smith progressive collapse, Navier plate field,
  yield-line capacity, ISO 19906 and CSR screening.
* `tensorfem.stiffened_panel_unit_fe` — elasto-plastic Shell4 unit cell and a
  stateful spherical arc-length (Riks) driver.
* `benchmarks/stiffened_panel_ultimate_strength.py` — the evidence run
  (closed-form chain + cached FE traces + figures).
* `tests/test_stiffened_panel_ultimate.py`, `tests/test_stiffened_panel_unit_fe.py`
  — regression tests (closed forms, rule checks, FE driver).

## 1. Case definition

| quantity | ice-belt panel | single-bay panel |
|---|---|---|
| span `a` (frame spacing) | 2.4 m | 0.8 m |
| stiffener spacing `b` | 0.800 m | 0.800 m |
| plate thickness `t_p` | 20 mm | 12 mm |
| web | 200 × 10 mm | 180 × 9 mm |
| flange | 80 × 12 mm | 80 × 12 mm |
| material | E = 206 GPa, ν = 0.3, σ_y = 235 MPa | same |
| plate slenderness β | 1.35 | 2.25 |
| column slenderness λ | 0.47 | 0.66 |

Both are idealised (no corrosion, no weld-induced residual stress in the FE
model, no openings). The 20 mm / 800 mm ice-belt geometry is the geometry named
in the campaign acceptance criteria; it is *not* taken from a specific vessel.

## 2. Method

### 2.1 Closed-form component chain

* **Plate ultimate** — Faulkner / von Karman effective width,
  `σ_u/σ_y = min(1, 2/β − 1/β²)`, β = (b/t)√(σ_y/E). Empirical fit, valid for
  β ≳ 1.
* **Stiffener column** — Johnson–Ostenfeld parabolic curve with the Euler stress
  of the stiffener plus attached plating, `σ_E = π²EI/(A a²)`.
* **Stiffener tripping** — St Venant torsion only,
  `σ_T = (GJ + π²EI_w/a²)/I_p`, warping from the flange
  (`I_w = t_f³ b_f³/144`). This screen ignores the in-plane restraint of the
  flange and the attached plating, so it is **reported but excluded** from the
  default limiting stress (`include_tripping=False`); for the ice-belt panel it
  gives 137 MPa, i.e. far below the column value of 222 MPa.
* **Local flange buckling** — one-edge-free plate, k = 0.425.
* **Panel ultimate** — area-weighted: `σ_u = (A_p σ_p + A_s σ_s)/(A_p + A_s)`.
* **Load-end shortening** — limiting-stress: elastic until the component limit,
  then a plateau (no post-ultimate softening).
* **Smith (1977) progressive collapse** — incremental curvature, per-unit
  limiting stress on the compression side, elasticity capped at σ_y on the
  tension side, neutral axis held at its elastic position.
* **Plate field under pressure** — exact Navier double series for a simply
  supported rectangular plate; the governing centre moment is the larger of
  M_x and M_y.
* **Rigid-plastic plate capacity** — kinematic upper bound. The four-hinge
  pyramid mechanism gives `p = 12 M_p (a² + b²)/(a²b²)` (24 M_p/a² for a square
  plate); the one-way strip gives 8 M_p/b² (16 M_p/b² clamped). The strip
  mechanism is only kinematically admissible for long fields
  (`a/b ≥ 2`, where the short-edge restraint is local), so the selector uses
  the pyramid for `a/b ≤ 1.5`, the strip for `a/b ≥ 2`, and the smaller value
  (flagged `transition`) in between.

### 2.2 Finite-element unit cell

Half-bay model: plate field `a × b/2` of layered Shell4 (5 layers through the
thickness is the production default; 3 layers are used here), a web strip and a
flange strip sharing nodes with the plate (rigid connection), 16–24 elements,
90–216 free DOFs. Thicknesses are per element group, so plate 20 mm / web
10 mm / flange 12 mm coexist.

Boundary conditions:

* `x = 0` (frame line): `u_x = 0`, `w = 0`; `x = a`: `w = 0` (loaded edge).
* `y = 0` (stiffener centreplane): `u_y = θ_x = θ_z = 0` (true symmetry).
* `y = b/2` (mid-bay): `θ_x = θ_z = 0` only. **Prescribing `u_y = 0` here as
  well suppresses the transverse contraction of the cell and locks the plating
  into biaxial compression**; in an earlier revision of this model that raised
  the apparent strength to 1.17 σ_y (a 25 % overestimate) with no limit point.
* Imperfections: column mode `a/1000 · sin(πx/a)` plus a plate mode
  `0.1 t_p · sin(πx/a) sin(πy/(b/2))`.

Loads: the reference compressive load is the tributary area times one pascal,
so the arc-length load factor **is** the applied average stress in pascals;
lateral pressure is a separate consistent-nodal-load field on the plate.

Driver: spherical arc-length (Crisfield) with
`g = |Δu|² + (w Δλ)² − Δs²`, consistent tangent from the layered J2 return
mapping, and an explicit material-state commit after every converged step. The
committed state cannot be frozen for a whole path — doing so converts the path
into a single plastic increment. The elastic limit of the driver is checked
against `tensorfem.arc_length.solve_arc_length` (relative agreement 1e-5 in the
test suite).

## 3. Acceptance results

Numbers below are written by
`benchmarks/stiffened_panel_ultimate_strength.py` into
`artifacts/stiffened_panel_ultimate/evidence.json`; the report body is
generated from that file, so no value is transcribed by hand.

### 3.1 Executed finite-element path (ice-belt panel, 24 arc steps)

| quantity | value |
|---|---|
| average stress at the last converged step | 0.9981 σ_y |
| closed-form area-weighted panel ultimate | 0.9345 σ_y |
| relative difference | **6.81 %** (< 10 % criterion) |
| maximum equivalent plastic strain reached | 1.863e-04 |
| path status | `approaching_limit_point` — increments decay 0.978→0.995→0.998 σ_y |
| limit point captured | **no** (honest FAIL of the "post-buckling limit point" criterion) |

The ice-belt plate is stocky: β = 1.35, so its elastic plate buckling stress is
1.98 σ_y and the section yields before it can buckle. The executed geometry is
therefore **yield governed** and has no limit point to capture; the arc-length
machinery is exercised, converging and approaching a limit, but the campaign's
"limit point + post-buckling branch" item is **not satisfied** by this run. The
slender single-bay geometry (β = 2.25) that would buckle was set up but its
arc-length run did not finish inside the compute budget, so it is not reported
as evidence — reporting a truncated path as an "ultimate strength" would be
wrong (its last load factor is only a lower bound).

### 3.2 Arc-length step independence

Two paths on the same ice-belt cell at Δs = 4.0e-4 m and Δs = 2.0e-4 m,
compared by the average stress at matched end shortening over their common
range: maximum relative difference **5.64e-15**. The two
paths coincide to round-off *because the load fraction per step halves with the
arc step*, so this demonstrates reproducibility of the trajectory rather than a
convergence rate; it does show that the path is not an artefact of the step
size at these resolutions.

### 3.3 Plate field under ice pressure (ISO 19906 chain)

| quantity | value |
|---|---|
| local ice pressure (pressure-area form, caller-supplied coefficients) | 493.4 kPa over 1.92 m² |
| elastic plate-field bending stress (Navier series) | 562.9 MPa |
| allowable stress (0.8 σ_y) utilisation | **2.40** (fails) |
| rigid-plastic capacity (yield-line, strip mechanism, a/b = 3.0) | 293.7 kPa |
| ULS utilisation p_ice / p_cap | **1.68** (fails) |
| allowable pressure implied by the chain (elastic / plastic) | 164.8 kPa / 293.7 kPa |
| plate moment coefficient M/(p b²) | 0.1189 (one-way field) |
| square-plate series coefficient vs Timoshenko 0.0479 | **0.028 %** |
| square-plate yield-line capacity vs 24 M_p/a² | **0.0 %** (identical by construction) |

The assumed representative ice pressure drives this idealised panel **beyond**
both its allowable stress and its plastic capacity. That is a statement about
the (assumed) pressure, not about a vessel: the chain is demonstrated on the
failing branch. The normative ISO 19906 allowable pressure is unavailable here,
so "allowable pressure within 5 % of the code value" is **WITHHELD**; the
verified 5 % items are the structural-side quantities above.

### 3.4 CSR hull-girder screening (Smith progressive collapse)

| quantity | value |
|---|---|
| neutral axis / section inertia | 3.745 m / 4.199 m⁴ |
| M_u (progressive collapse) | 240.8 MN·m |
| fully plastic bound M_p | 253.4 MN·m |
| applied design moment (0.55 M_p) | 139.4 MN·m |
| utilisation with γ_R = 1.05 | **0.608** (passes) |

### 3.5 Ice + wave combination

In-plane utilisation 0.55,
bending utilisation 2.40,
combined quadratic utilisation **2.46**
(fails — dominated by the
assumed ice pressure of §3.3).

### 3.6 Criterion status

```json
{
 "fe_matches_closed_form_within_10_percent": true,
 "arc_length_step_convergence_within_15_percent": true,
 "collapse_path_shows_peak": false,
 "fe_approaches_a_limit_point": true,
 "plate_field_series_within_5_percent_of_textbook": true,
 "yield_line_square_matches_24Mp_over_a2": true,
 "smith_1977_experimental_comparison": "WITHHELD",
 "abaqus_riks_comparison": "WITHHELD",
 "iso19906_normative_allowable_pressure": "WITHHELD"
}
```

## 4. Limitations

1. **No experimental validation.** The campaign asks for < 10 % against the
   Smith (1977) stiffened-panel tests. Those data are not in this environment,
   so the FE/closed-form agreement and the step-size reproducibility are the
   only evidence offered, and the criterion is marked WITHHELD rather than
   claimed.
1b. **The post-buckling limit point was not captured.** Only the stocky
   ice-belt cell (β = 1.35) completed; it is yield governed and has no limit
   point. The slender β = 2.25 cell was built and launched but its path did not
   complete in the budget, so the "limit point + post-peak branch" item is an
   honest FAIL for this node.
2. **Coarse through-width mesh.** A bay of 0.8 m with 4 elements across the half
   width resolves the first plate buckle mode only marginally; a coarse
   (`n_y = 2`) mesh cannot resolve it at all and returns a squash-dominated
   strength ≈ 1.1–1.2 σ_y. Those coarse results are retained in the evidence as
   a mesh-sensitivity warning, not as strength predictions.
3. **Limiting-stress curves have no softening**, so the hull-girder
   moment-curvature saturates on a plateau instead of dropping; the peak is
   therefore an upper estimate for a section whose units all reach their limit
   simultaneously.
4. **Tripping, weld residual stress, corrosion and openings are not modelled.**
   The tripping screen in particular is not a rule clause.
5. **All rule coefficients are caller supplied.** The ISO 19906 pressure-area
   coefficients used in the benchmark (1.5 MPa at 0.5 m², exponent −0.3) are
   *assumed representative values*, not values read from the standard; the CSR
   resistance factor is exposed as a parameter (1.05 in the benchmark) and must
   be set from the applicable edition.
6. **Interaction surfaces are screening assumptions.** The ice + wave
   combination uses a quadratic utilisation surface; it is not a class-society
   interaction equation.
7. The through-thickness layer count is 3–5; a plasticity localisation study
   would need more.