# FSI3D admission: BLOCKED

The `tensorfem.fsi_ice_3d_coupling.FSI3DIceCouplingSolver` prototype is
unavailable. Construction and all execution/diagnostic methods raise
`FSI3DUnavailableError` (a `NotImplementedError`). Its old calculation used
stiffness in place of mass, never advanced fluid variables or ice damage,
and replaced NaN energy with zero. Its mass residual was an unconditional zero.
Those outputs and earlier C1–C5 PASS claims must not be used as validation.

`PYTHONPATH=src python validate_fsi_ice_3d.py --output status.json` now emits a
machine-readable `blocked`, `passed: false` record and exits **2**. It does not
generate simulated curves or validation plots. No bypass enables the old path.

Configuration, shape functions, elastic element stiffness and the yield-surface
predicate remain available as experimental local helpers. Generated HEX8 node
ordering is corrected and inverted Jacobians are rejected. The unqualified
plastic return mapping now rejects yielded states. Local helper tests are not
ice fracture or FSI validation. This module is not the separate SUBOFF example.

Reopening requires all of the following, with actual computed evidence:

1. Consistent mass, displacement/velocity/acceleration integration, and a dry
   structural dynamics benchmark with time refinement.
2. Constitutive updates checked against independent loading/unloading references;
   objective damage evolution, energy dissipation, and mesh sensitivity.
3. A fluid solver integrated through a public, versioned interface; conservative
   traction transfer and kinematic matching at a moving interface.
4. Mass residual computed from discrete fluxes and geometry; kinetic energy
   from mass; external work, strain energy and dissipation balance.
5. Coupling convergence, spatial/time refinement, and independent ship–ice
   reference data with calibration/validation sets separated.

These are TensorFEM's solver evidence requirements. Generic orchestration stays
outside this repository; passing software tests does not reopen the physics gate.
