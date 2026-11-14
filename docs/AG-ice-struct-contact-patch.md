# GAP-PP-1-ICE-STRUCT-CONTACT-PATCH — ice patch load from a CZM contact solve, fed two-way into the panel solver

Status: implemented + verified (design point and uniform-limit checks), full-resolution
scan submitted to the cluster (job `cj2535`) and not yet returned — see "Open items".

## What was missing

`papers/jour-ice-struct/code/panel_patch_load.py` (result file
`data/panel_patch_load_results.json`) treats the ice pressure patch as a *given
uniform* pressure.  The library could solve an ice-structure cohesive interface
(`tensorfem.cohesive.BilinearCohesiveLaw`) but nothing projected a contact
pressure field onto a plate load vector, and nothing fed the panel deflection
back into the contact opening.  This change adds exactly that chain.

## What was added

`src/tensorfem/ice_panel_patch_contact.py`

| piece | content |
|---|---|
| `czm_contact_patch` | local ice-structure normal contact: opening `g = delta0 - s(x,y) - w(x,y)`, pressure `p = max(0, -t_cohesive(g))` capped by the cohesive peak traction (crushing ceiling); bisection on the rigid approach `delta0` enforces `∫p dA = F` |
| `patch_from_contact` | conservative projection of the pressure field onto the panel nodal load vector by **exact sub-cell overlap integration** of `∫N_i p dA` (panel element split at contact-grid lines; 2x2 Gauss exact for bilinear x bilinear) |
| `solve_rect_panel` | rectangular `[0,a]x[0,b]` Q4 Mindlin panel through the **merged** batched path (`uniform_rect_mesh`, `assemble_plate_foundation_system`, `solve_assembled_system`) — no new element/assembly code |
| `coupled_patch_solve` | `contact -> load -> plate -> deflection -> contact` fixed-point loop, force held at every sweep |
| `uniform_patch_pressure` | uniform reference patch of identical force and footprint (the JOUR-ICE-STRUCT idealisation, same caliber) |
| `navier_rect_moments` | independent Kirchhoff Navier series for a **non-uniform** pressure field on a rectangular SSSS plate (Fourier coefficients by Gauss quadrature of the same field) |
| `nonuniformity_scan` | four contact configurations, each against a uniform patch of equal force and footprint, machine-readable JSON |
| `design_point_calibration` | design-point reproduction of the published baseline + Timoshenko Table 8 caliber check |

`benchmarks/ice_panel_patch_contact_scan.py` runs both studies and writes
`outputs/ice_panel_patch_contact_scan.json` plus two figures.
`tests/test_ice_panel_patch_contact.py` holds 12 tests.

## Verification (measured, not asserted)

Machine: loong64 dev node (load15 > 100 at the time, so only light local runs;
the full-resolution scan went to the scheduler).

Panel solver vs the tabulated reference — SSSS square 1 m, t = 22 mm, E = 206 GPa,
nu = 0.3, uniform 1 MPa, mesh 40x40:

| quantity | FEM | reference | deviation |
|---|---|---|---|
| `M_max/(q a^2)` | 0.0478161 | 0.0479 (Timoshenko Table 8, rounded) | **-0.175 %** |
| `M_max/(q a^2)` | 0.0478161 | 0.0478864 (exact Navier series) | **-0.147 %** |
| `w_max D/(q a^4)` | 0.0040717 | 0.00406 | +0.29 % |

Independent series check: the same series, driven by a uniform full-plate patch,
returns `M_max/(q a^2) = 0.04788641` — the closed-form value, so the reference
implementation is itself anchored.

Design point of `panel_patch_load_results.json` (`lambda_c/l = 0.1`,
`full_breadth`, uniform 1 MPa over 0.08 m x 0.7 m, mesh 40x35):

| quantity | this FEM path | Navier series (this module) | published baseline |
|---|---|---|---|
| `sigma_Mx` [MPa] | 79.087 | 79.590 | 78.668 |
| `sigma_vm` [MPa] | 73.975 | 74.360 | 73.371 |

* FEM vs independent series: `-0.63 %` (sigma_Mx), `-0.52 %` (sigma_vm)
* FEM vs published baseline (Kirchhoff Navier in the paper repo): `+0.53 %`
  (sigma_Mx), `+0.82 %` (sigma_vm) — i.e. **sub-percent, the same caliber as the
  H3-ICE-PLATE-FEM anchor (0.339 %)**
* force projection error: `7.6e-16` relative (exact by construction)

Chain self-check: a flat ice edge with no plate coupling produces a pressure
field whose peak/mean ratio is 1 to machine precision and whose panel stress
equals the uniform-patch reference to `1e-9` relative — the projection and the
solver cannot both be wrong in a way that cancels here.

`pytest tests/test_ice_panel_patch_contact.py` → 12 passed (uniform/Timoshenko,
Navier anchor, force conservation on a non-aligned grid, force target and
ceiling for three loads, monotone contact area, flat-edge uniform limit, two-way
coupling convergence + force preservation, FEM vs series for a non-uniform
patch, scan schema with >= 3 configurations and C1 ratio = 1).

## Where the numbers that matter come from

The whole point of the anchor is the *non-uniformity delta*: peak panel stress
from a CZM-generated pressure field versus the uniform patch of the same force.
The scan machinery is verified by the tests above (C1 flat-edge collapses to
ratio 1; curved configurations deviate by more than 1 %), but the full-resolution
table (nx=40, ny=35, contact grid 97) was submitted as cluster job `cj2535` and
had not returned when this note was written, so no full-resolution delta is
quoted here.  Nothing is extrapolated.

## Notes, limits, pitfalls

* The interface law is used only through its undamaged normal (penalty) branch;
  the peak traction is used as a perfectly plastic crushing ceiling and the mode-I
  softening branch is deliberately not activated in compression (declared
  deviation, documented in the module docstring).
* Ice-side elasticity is lumped into `k_n = E_crush / h_layer`; no ice continuum
  is solved.
* `panel_patch_load_results.json` compares against a *Kirchhoff* Navier solution;
  this path is Mindlin-MITC, so the residual +0.5 % is partly the shear
  deformation and partly the different quadrature — both are sub-percent, which is
  what the acceptance asks for.
* Pitfall found while building this: the von Mises moment at element centres
  carries a spurious `Mxy` boundary layer next to the simply supported edges of a
  Q4 Mindlin mesh (up to ~17 % above the centre value).  `PanelSolution.peak_stress`
  therefore excludes the outer element ring by default and `stress_at` is used
  for the baseline-caliber sampling.
* Pitfall found: the series was initially summed without its `D` factor and
  without `sin(k_m x)`, giving a 2e5 error on a uniform plate — caught by the
  uniform-square regression, which is why that test exists.
* Pitfall found: an unaligned uniform-patch grid discretises the rectangle to
  52.5 kN instead of 56 kN (-6.3 %).  `uniform_patch_pressure` now inserts the
  patch edges into the contact grid so the sampled field is exact.

## Open items

1. Full-resolution scan table (job `cj2535`): figures `fig1_czm_contact_pressure_patches.png`,
   `fig2_peak_stress_nonuniform_vs_uniform.png` and
   `outputs/ice_panel_patch_contact_scan.json`.
2. `patch_from_contact` is the runtime bottleneck in the coupling loop (pure
   Python sub-cell loop); a vectorised version is a straightforward follow-up.
3. Out of scope: plasticity/buckling of the panel under the patch, ice continuum
   constitutive model, HPZ (high-pressure-zone) statistics.
