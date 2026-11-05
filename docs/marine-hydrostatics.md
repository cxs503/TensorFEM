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

### Bounded validity of the linear righting arm (audited)

`benchmarks/marine_large_angle_stability_audit.py` rebuilds the exact
wall-sided righting arm from immersed-section geometry and measures the
`GM*sin(phi)` arm against it. The sine arm stays inside 3% at the declared
10 degrees only while `GM >= 0.5027*BM` (see
[the audit](marine-large-angle-stability-audit.md)); the two example cases in
`examples/marine/` satisfy that, a low-`GM` barge of the same geometry reaches
3.27% at 10 degrees and the exact arm already differs from the linear one at
2.09% for the shipped proportions. The audit also bounds the closed form
`sin(phi)*(GM + BM*tan^2(phi)/2)` to `min(phi_deck, atan(2*T/B))`.

## Deliberate limitations

This module still is not a large-angle `GZ` curve; the exact wall-sided `GZ`
oracle lives in `tensorfem.marine_large_angle_stability` and is documented in
[the audit report](marine-large-angle-stability-audit.md). It is not a
large-angle `GZ` curve, cross-curves, free-surface, damage
stability, compartment flooding, trim, arbitrary-hull integration, wave-load,
seakeeping, mooring, or classification-rule implementation. Passing its tests
does not constitute design approval or regulatory/classification certification.
