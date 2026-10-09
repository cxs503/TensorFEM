# Continuous icebreaker simulation: reuse the SUBOFF case and select a real icebreaker hull

## Decision summary

Keep the existing SUBOFF upward-ice-impact case as a **structural/contact verification fixture**, but do not use the submarine as the geometry for a continuous icebreaker-resistance study. For the first ship-shaped model, use the U.S. Coast Guard icebreaker **USCGC Glacier (WAGB-4)** because its Historic American Engineering Record (HAER CA-341) archive includes a measured lines plan, hull sections, profile and arrangement drawings. The U.S. Coast Guard fact sheet provides principal dimensions.

Sources:
- Lines plan, HAER CA-341 sheet 6: https://www.loc.gov/resource/hhh.ca3458.sheet/?sp=6
- Library of Congress survey and rights notes: https://www.loc.gov/item/ca3458/
- Official USCG dimensions and service facts: https://www.history.uscg.mil/Browse-by-Topic/Assets/Water/All/Cutters-65-ft-or-greater/Article/2101538/glacier-1966-wagb-4/

The lines plan is a raster engineering drawing, not a ready-to-run CAD surface. Digitization, scale calibration, station alignment and geometry audit are still required; no exact hull surface should be claimed until that extraction is completed and checked against the drawings.

## Why keep SUBOFF, but not as the icebreaker

The existing `suboff_upward_ice_collision.py` and `suboff_appended_ice_simulation.py` cases already exercise a closed shell, hull/ice contact, cohesive ice seams, fragment-connectivity diagnostics, energy accounting and repeatable output. They are valuable for regression tests and structural/contact coupling.

Their published reports explicitly limit the result to short-duration, upward impact; the physical accuracy is not qualified, and mesh/contact sensitivities remain above the declared 3% acceptance threshold in key metrics. They do not represent forward travel, a purpose-designed icebreaking bow, propulsive resistance or fully coupled LBM water flow. Preserve these limitations in every report.

## Candidate ship model: USCGC Glacier

The USCGC fact sheet lists:
- Overall length: 309 ft 8 in = approximately 94.39 m
- Beam: 74 ft 4 in = approximately 22.66 m
- Draft: 28 ft 3 in = approximately 8.61 m
- Displacement at commissioning: 8,650 tons

For an initial geometry and solver-development fixture, a **1:20 geometric scale** gives approximately:
- Length: 4.719 m
- Beam: 1.133 m
- Draft: 0.431 m

This scale is a numerical/model-geometry choice, not a claim of complete Froude, Reynolds or ice-fracture similitude. Select model speed and ice thickness only after documenting the intended similarity rules and available benchmark data.

Use the HAER body plan, half-breadths, sheer plan and frame sections to reconstruct the hull. The measured drawings are associated with the U.S. National Park Service/Library of Congress survey; the Library of Congress notes that there are no known restrictions on U.S. Government-created images, while warning that some copied material may have separate restrictions. Record the exact source sheet, calibration method, derived geometry and provenance in the repository.

## Architecture: three roles, one coupling driver

- **TensorFEM**: retain the SUBOFF upward-impact case for shell/ice contact regression. Later, optionally use FEM for local hull-shell stress checks. Do not make FEM's current cohesive ice tiles the production ice model.
- **TensorDEM**: own bonded-particle ice, irreversible bond failure, post-failure contact and floe tracking. The current 2-D DEM is a prototype; add rotational degrees of freedom, tangential bond/contact mechanics and calibrated failure parameters before engineering resistance claims.
- **TensorLBM**: own water/air flow and hydrodynamic force reconstruction. Keep lattice-unit conversion at the solver boundary; separately record water force on hull and water force on ice.
- **Coupling driver**: own the common hull trajectory, synchronization/subcycling, load transfer, SI-unit exchange, force/moment checks, resistance ledger and run metadata.

Do not transfer the FEM cohesive-ice force into DEM as well; that would double-count the ice mechanics. Use the FEM case as a separate regression benchmark unless a deliberate hybrid region is defined.

## Development sequence

### Stage A — reproduce and freeze the existing SUBOFF baseline

1. Run both documented SUBOFF examples and the independent evidence validators on the active `master` branch.
2. Store the exact source commit, configuration, geometry hashes and solver outputs with each run.
3. Keep time-step sensitivity, mesh sensitivity and physical-validation status as separate fields. Existing conservation checks verify numerical bookkeeping, not real-world accuracy.

### Stage B — extract a Glacier bow geometry

1. Digitize station curves from the HAER lines plan and frame-section sheet; calibrate scale using drawing dimensions and cross-check the principal dimensions.
2. Reconstruct a watertight surface or a documented longitudinal 2-D bow profile. Keep the 2-D profile labelled as a reduced model, because it cannot reproduce three-dimensional bow flare and lateral ice clearing.
3. Export a versioned geometry file with source-sheet URL, unit convention, scale ratio, station coordinates, validation checks and geometry checksum.
4. Audit length, beam, draft, waterline, symmetry, surface normals, watertightness and displacement volume before using the geometry in a force calculation.

### Stage C — prescribed-speed DEM icebreaking, without fluid

1. Start with a longitudinal 2-D bow/ice interaction and prescribed forward speed; do not begin with free 6-DOF ship motion.
2. Generate a bonded-particle ice sheet with documented particle spacing, bond normal/shear law, failure energy, ice–hull friction and boundary treatment.
3. Move the hull through the ice and log longitudinal ice-contact resistance, vertical force, pitch moment, bond-break events, floe sizes and energy balance.
4. Check particle-resolution and time-step sensitivity, and calibrate the ice parameters against an identified ice-indentation or model-scale icebreaking reference. Do not tune only to a single target resistance curve.

### Stage D — add TensorLBM in one-way, then conservative two-way coupling

1. First calculate the fluid-only resistance of the prescribed hull in open water and validate force integration independently.
2. Add the DEM ice field and transfer fluid traction to ice particles only after mapping and area/thickness conventions are explicit.
3. At every coupling interval compare force and moment resultants across the fluid/ice interface and test equal-and-opposite exchange. Keep fluid-on-hull, ice-contact-on-hull and fluid-on-ice as separate ledger entries.
4. Run open-water, zero-traction, no-ice, rigid-ice and broken-ice controls before interpreting combined resistance.
5. Only after the 2-D contract passes should the project advance to 3-D hull geometry and free-surface LBM.

## Minimum outputs for the first useful ship case

- `geometry_manifest.json`: source drawings, scale, dimensions, units, checksum and geometry-audit results.
- `run_config.json`: speed, ice thickness, particle spacing, material calibration ID, LBM resolution and coupling interval.
- `resistance_history.csv`: time, hull position, fluid resistance, DEM ice-contact resistance, total resistance, vertical force and pitch moment.
- `fracture_events.csv`: time, bond/particle IDs, failure mode, dissipated energy and floe/component ID.
- `balance.json`: mass, momentum, force/moment transfer mismatch, energy residual and all acceptance thresholds.
- A machine-readable status that distinguishes `numerically_consistent`, `resolution_sensitive` and `physically_validated`.

## Repository note

At the time of writing, the detailed TensorFEM implementation and the SUBOFF case are on the `master` branch, while the repository default branch `main` contains only the short project README. Keep this roadmap targeted at `master` so it stays alongside the implementation; reconcile branch/default-branch policy separately rather than silently transplanting or replacing the code tree.
