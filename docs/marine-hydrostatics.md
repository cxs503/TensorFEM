# Marine box-barge hydrostatics

`tensorfem.marine_hydrostatics` provides a small, auditable first step for ship
and offshore floating-structure applications. It evaluates an upright,
rectangular wall-sided barge in SI units.

## Supported calculations

- displaced volume and mass;
- center of buoyancy (`KB = T/2`);
- transverse waterplane second moment (`I = L B^3/12`);
- transverse metacentric radius and height (`BM = I/volume`, `GM = KB + BM - KG`);
- righting arm and restoring moment at heel angles up to 10 degrees;
- either specified draft/KG or equilibrium draft/KG assembled from lightship
  and named load items.

The two inputs in `examples/marine/` show both modes. `load_case` conserves
mass and evaluates the combined vertical center of gravity. Inputs reject
unknown keys, non-SI units, invalid geometry, overload/submergence, and heel
angles outside the supported range.

## Verification basis

The implementation follows the closed-form rectangular-prism hydrostatic
relations above. The regression oracle for restoring moment independently uses
the linear small-angle relation `M = displacement * g * GM * phi`; the sine
form implemented by the solver differs by less than 3% over the declared
range. Tests also verify mass balance, load-case KG, deterministic reporting,
and fail-closed validation.

## Deliberate limitations

This is not a large-angle `GZ` curve, cross-curves, free-surface, damage
stability, compartment flooding, trim, arbitrary-hull integration, wave-load,
seakeeping, mooring, or classification-rule implementation. Passing its tests
does not constitute design approval or regulatory/classification certification.
