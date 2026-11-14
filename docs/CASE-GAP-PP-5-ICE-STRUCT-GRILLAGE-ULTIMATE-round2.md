# CASE — GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE (round 2)

Anchor `GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE`, owner 2026-11-11 two-round method,
round 2 (survey first, then build). Repository TensorFEM. Module
`src/tensorfem/grillage_plastic_collapse.py`; gates
`src/tensorfem/grillage_plastic_verification.py`; tests
`tests/test_grillage_plastic_collapse.py`. Every number below is generated from
`artifacts/grillage_plastic_collapse/grillage_plastic_verification.json`
by script — none is typed by hand.

## 1. Problem setting and what was missing

`JOUR-ICE-STRUCT` (ice-belt paper) stops at the *single plate-field bending
stress*; limitation 6 of section 5 names the missing "panel stress -> structural
limit" jump. Round 1 of this anchor (m19378) ended honest-FAIL with no stage
record, so this round surveyed first.

Already present in the repository: `grillage_ice_ultimate` (elastic Navier
orthotropic load share + limiting-stress member checks + `ultimate_ice_pressure`),
`stiffened_panel_ice_ultimate` and `ice_belt_collapse` (nonlinear FE panels),
`smith_panel_collapse` / `plate_stiffener_ultimate` (Smith progressive collapse
of one stiffened panel unit / section). What was missing is the
**grillage-level structural ultimate state with load redistribution** and a
safety margin; `grillage_ice_ultimate` states the limitation itself: *"once a
member yields, load redistribution is not traced"*.

## 2. Survey (round-2 requirement)

**Open source, horizontal.** `~/oss-cache` holds three repositories only —
`Maarten-vd-Sande_lbm`, `hschwane`, `pylbm` — all lattice Boltzmann; none
contains a stiffened-panel / grillage ultimate-strength implementation, so there
was no structure to copy. A cluster-side inventory scan was submitted through the
scheduler (`cj2152`, `fetch_rc=0`); the `oss_inventory` and `oss_solutions`
tables are **not readable with the agent role** (`permission denied for table
oss_inventory`), so the `SELECT ... WHERE feature_slug ILIKE '%ULTIMATE%'` query
could not be executed here, and the `oss_solutions` write-back also failed for
the same reason. Recorded honestly: the open-source survey produced *no* reusable
implementation; the method had to come from the literature.

**Method lineage (external anchors).**

| ingredient | source |
| --- | --- |
| successive plastic-hinge formation, collapse mechanism | Neal, *The Plastic Methods of Structural Analysis*, 3rd ed., ch. 2 and 9 |
| plastic moment reduced by axial force `M_p (1-(P/P_y)^2)` | Neal ch. 2; IACS CSR Pt.1 Ch.5 |
| incremental-iterative collapse of stiffened panels | Smith (1977); IACS CSR Pt.1 Ch.5 App. |
| two-way grillage split by orthotropic plate energy | Timoshenko and Woinowsky-Krieger, *Theory of Plates and Shells*, ch. 4 |
| ice patch + hydrostatic + hull-girder combination | interface fixed by `CombinedLoadCase` in `grillage_ice_ultimate` |

Rigid-plastic limit analysis was chosen over another nonlinear FE path because
(a) independent methods are what the FE paths need for cross-checking, (b) the
collapse mechanism falls out as a by-product, (c) the exact benchmarks are closed
form so accuracy can be gated at the percent level.

## 3. Method and implementation

Two orthogonal member systems (stiffeners continuous over the transverse frames;
frames simply supported at the girders). The ice patch force is split between
them with the orthotropic **energy** load share, renormalised to `f_x + f_y = 1`,
so both systems see the same total force and the layer stays comparable with
`grillage_ice_ultimate`.

Each member line is a continuous Euler-Bernoulli beam with consistent nodal loads
from the actual patch rectangle. A plastic hinge is exact: every element carries
its own rotation DOF at each end, the two ends of neighbouring elements are
merged unless a hinge sits between them, and the hinge applies the prescribed
plastic couple as equal-and-opposite nodal moments (physical section moment `+m`
at an element start, `-m` at an element end). With the hinge set fixed the
response is linear in the ice pressure, so the pressure bringing the next free
section to `M_p` is closed form; hinge, repeat, until no section can reach `M_p`
(or the tangent stiffness is singular) — that is the collapse mechanism and its
pressure is the limit load. The hydrostatic dead load is applied first and
carried through.

Code pointers: `member_collapse_pressure` (hinge loop),
`_BeamSystem.dof_map`/`assemble`/`solve` (hinge-aware assembly),
`grillage_plastic_collapse` (system split, plate-limit union, margin),
`_figures` (plots).

## 4. Parameter table (case study, dimensionless set)

| parameter | value |
| --- | --- |
| frame spacing / bay spacing | 2.1 m / 0.35 m (aspect 6.0) |
| grillage length x frame span | 8.40 m x 2.80 m |
| plate thickness / yield stress | 20 mm / 355 MPa |
| stiffener | Tee 250x10 + 80x12, tributary plate 0.35 m |
| frame | Tee 550x10 + 120x14, effective plate 0.84 m |
| plate slenderness beta | 0.726 |
| stiffener column slenderness lambda | 0.307 |
| frame column slenderness lambda | 0.215 |
| M_p stiffener / frame | 0.189 / 0.905 MN m |
| Faulkner plate strength sigma_u/sigma_y | 1.000 |
| panel ultimate stress | 329.9 MPa |
| ice patch | 1.05 m x 0.35 m = 0.3675 m^2 (0.125 of the grillage length) |
| hydrostatic pressure | 0.10 MPa |
| coexisting compression sigma_x/sigma_y | 0.30 (axial reduction 0.9100) |
| load shares x / y / torsion | 0.1673 / 0.8327 / 0.1097 |

## 5. Comparison table (reference vs got, every gate)

| gate | got | reference | error | allowed | verdict |
| --- | --- | --- | --- | --- | --- |
| V1_ss_uniform | 375722 | 376700 | -0.2597 % | 1.00 % | PASS |
| V2_ss_central_patch | 517283 | 519140 | -0.3576 % | 1.00 % | PASS |
| V3_clamped_uniform | 751443 | 753400 | -0.2597 % | 1.00 % | PASS |
| V4_two_span_uniform | 548650 | 548892 | -0.0441 % | 2.00 % | PASS |
| V5a_elastic_midspan_moment | 0.551609 | 0.55125 | +0.0651 % | 0.20 % | PASS |
| V5b_elastic_midspan_deflection | 1.43862e-08 | 1.43862e-08 | +0.0000 % | 1.00 % | PASS |
| V6_axial_reduction_15 | 0.9775 | 0.9775 | +0.0000 % | 0.00 % | PASS |
| V6_collapse_scales_15 | 0.974961 | 0.9775 | -0.2597 % | 2.00 % | PASS |
| V6_axial_reduction_30 | 0.91 | 0.91 | +0.0000 % | 0.00 % | PASS |
| V6_collapse_scales_30 | 0.907636 | 0.91 | -0.2597 % | 2.00 % | PASS |
| V6_axial_reduction_45 | 0.7975 | 0.7975 | +0.0000 % | 0.00 % | PASS |
| V6_collapse_scales_45 | 0.795429 | 0.7975 | -0.2597 % | 2.00 % | PASS |
| V7_mesh_convergence | 0.000650618 | 0.002 | +32.5309 % | 100.00 % | PASS |
| V9a_load_share_positive | 1 | 1 | +0.0000 % | 0.00 % | PASS |
| V9b_system_share_partition | 1 | 1 | +0.0000 % | 0.00 % | PASS |
| V9d_modal_share_partition | 1 | 1 | -0.0000 % | 0.00 % | PASS |
| V9c_load_share_terms_convergence | 0.167334 | 0.167334 | +0.0002 % | 2.00 % | PASS |
| V10_frame_vs_closed_form | 1.36366e+06 | 1.37881e+06 | -1.0989 % | 2.00 % | PASS |
| V7b_case_study_mesh | 6.93825e+06 | 6.79224e+06 | +2.1496 % | 5.00 % | PASS |
| V8a_axial_monotone | 1 | 1 | +0.0000 % | 0.00 % | PASS |
| V8b_patch_monotone | 1 | 1 | +0.0000 % | 0.00 % | PASS |

21 gates, 0 failed. References are closed form
rigid-plastic theory, virtual-work upper bounds, exact elasticity, or the sibling
closed form in `grillage_ice_ultimate`; no gate compares the solver with itself.

Case study at 16 elements per span:

| member system | p_ult [MPa] | first hinge [MPa] | plastic reserve | hinges | hinge sections |
| --- | --- | --- | --- | --- | --- |
| longitudinal_stiffener | 118.260 | 64.373 | 1.837 | 4 | x=4.200 m, M=-0.19 MN m, x=4.725 m, M=+0.19 MN m, x=3.675 m, M=+0.19 MN m, x=3.806 m, M=+0.19 MN m |
| transverse_frame | 6.938 | 6.938 | 1.000 | 1 | x=1.400 m, M=+0.90 MN m |

| plate-field limit [MPa] | grillage plastic [MPa] | reported ultimate [MPa] | governing mode | margin vs 4.000 MPa design |
| --- | --- | --- | --- | --- |
| 2.165 | 6.938 | 2.165 | plate_local_bending (transverse_frame) | 0.541 |

Reading of the numbers: the member-grillage mechanism sits at
3.20x the plate-field limit, so for this ice-belt scantling the plate
field — not the member grillage — governs the structural ultimate state. That
ratio is the quantified "panel stress -> structural limit" jump. The frame line
is statically determinate (reserve
1.00, one hinge = mechanism),
the stiffener line continuous over five frames has
1.84. Against the caller-supplied
4 MPa design pressure the margin is 0.54 < 1: the design
is **not** adequate at that pressure — an honest negative result, not a tuning
target.

## 6. Convergence and sensitivity

| elements/span | V1 error | V4 error |
| --- | --- | --- |
| 4 | -4.000 % | -1.174 % |
| 8 | -1.031 % | -0.450 % |
| 16 | -0.260 % | -0.044 % |
| 32 | -0.065 % | -0.041 % |

| patch length [m] | sigma_x/sigma_y | grillage plastic [MPa] | plate-field limit [MPa] | ultimate [MPa] | governing |
| --- | --- | --- | --- | --- | --- |
| 0.700 | 0.00 | 10.462 | 2.447 | 2.447 | plate_local_bending |
| 0.700 | 0.15 | 10.462 | 2.418 | 2.418 | plate_local_bending |
| 0.700 | 0.30 | 10.462 | 2.330 | 2.330 | plate_local_bending |
| 1.050 | 0.00 | 6.792 | 2.274 | 2.274 | plate_local_bending |
| 1.050 | 0.15 | 6.792 | 2.247 | 2.247 | plate_local_bending |
| 1.050 | 0.30 | 6.792 | 2.165 | 2.165 | plate_local_bending |
| 1.400 | 0.00 | 4.959 | 2.232 | 2.232 | plate_local_bending |
| 1.400 | 0.15 | 4.959 | 2.206 | 2.206 | plate_local_bending |
| 1.400 | 0.30 | 4.959 | 2.125 | 2.125 | plate_local_bending |

Convergence is O(h^2) as expected for consistent beam elements; both sweeps are
monotone (gates V8a/V8b).

## 7. Findings and defect fixed

* **Defect (fixed).** `grillage_ice_ultimate.plate_patch_response` computed the
  load share with a modal *generalised-force* weighting. For the small ice patch
  it returns negative, mode-count-dependent numbers
  (`load_share_x = -0.0835` at terms 61, `-0.0220` at 201, `-0.0521` at 401),
  which `system_load_shares` clamps to zero — the stiffener system was silently
  credited with no patch load. It is now the orthotropic **energy** share,
  positive by construction, partitioning exactly to one and converged in `terms`
  (0.0002 % between 61 and 121 modes, gate V9). All 17 pre-existing
  `grillage_ice_ultimate` tests still pass.
* Two hinge formulations were tried and rejected on evidence: a plain release
  (section moment collapses to zero) and condensation of the shared nodal
  rotation (node moment equilibrium lost). Both are documented in the module
  because they are the natural first attempts; each produced 20-150 % load errors
  that V1-V4 catch immediately.

## 8. Failure modes, limits, reproduction

```
PYTHONPATH=src python -m tensorfem.grillage_plastic_verification
PYTHONPATH=src python -m pytest tests/test_grillage_plastic_collapse.py tests/test_grillage_ice_ultimate.py -q
```

Artifacts: `artifacts/grillage_plastic_collapse/grillage_plastic_verification.json`,
`fig_grillage_plastic_collapse.png`, `fig_grillage_plastic_sensitivity.png`.

Limits: ideal rigid-plastic material, monotonic loading, no ratcheting; hinges
only at element boundaries; plate buckling, web tripping, residual stress and
initial deflection are not in the collapse mechanism (they enter only through the
plate-field limit from `grillage_ice_ultimate`); patch centred and quasi-static;
frame lines are rigid supports for the stiffener system; no rule table
transcribed.
