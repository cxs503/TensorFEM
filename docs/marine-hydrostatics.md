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

## Beyond 10 degrees

`tensorfem.marine_large_angle_stability` extends the *rectangular wall-sided*
case to the whole upright range with the exact closed form
`GZ = sin(phi) * (GM + 0.5 * BM * tan^2(phi))`, cross-checked against an
independent polygon/shoelace oracle (`y_B = BM tan(phi)`,
`KB' = T/2 + 0.5 BM tan^2(phi)`) and the angle-of-loll identity
`tan^2(phi) = -2 GM / BM`. Its audit also quantifies the error of the
`GM * sin(phi)` model used here: within the declared 10 degrees that error is
1.9-2.0% for `BM/GM ~ 1.2`, but it reaches 15.0% at 10 degrees when
`BM/GM ~ 11` (wide, shallow, lightly loaded barge, `KG` close to `KM`), so the
10-degree cap alone does not bound the righting-arm error at 3%. See
[marine-large-angle-stability-verification.md](marine-large-angle-stability-verification.md).
The extension is still restricted to rectangular wall-sided sections; real hull
forms need cross-curves.
