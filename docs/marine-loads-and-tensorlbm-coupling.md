# Marine environmental loads and TensorLBM coupling

This package supplies an auditable **load generator**, not a CFD solver. It covers
linear Airy-wave kinematics and the inline Morison equation for a fixed, slender,
vertical circular member. TensorLBM remains responsible for resolved flow fields,
free-surface CFD, vortex shedding and higher-fidelity fluid physics.

## Supported engineering scope

- finite-depth Airy dispersion and velocity/acceleration kinematics;
- uniform collinear current combined with wave velocity in the drag term;
- drag plus inertia force per unit length;
- base shear and seabed overturning moment by Gauss-Legendre integration;
- a closed-form quarter-phase oracle for regression and coupling calibration.

Coordinates use still-water level `z=0` and seabed `z=-depth`. All case inputs are
explicit SI. Unknown fields, unit substitutions, invalid coefficients, out-of-water
kinematics, malformed arrays and non-monotone time histories fail closed.

The supplied oracle integrates the inertia distribution analytically. At wave phase
`pi/2`, wave horizontal velocity is zero; a uniform current therefore also gives an
exact constant drag integral. Tests require both base actions to agree within 3%.

## TensorLBM to TensorFEM exchange boundary

`tensor-solver.load-history.v1` is intentionally narrow. TensorLBM exports strictly
increasing times, unique target FEM node IDs, and SI nodal forces shaped
`[time, node, 3]`. TensorFEM validates the complete document before returning tensor
data. Node mapping/conservative surface projection is a preprocessing responsibility;
the adapter does not silently guess a mapping. `force_history_resultants` provides an
independent force/moment audit after mapping.

See [the force-history example](../examples/marine/tensorlbm_force_history.json) and
[the Airy/Morison case](../examples/marine/fixed_pile_airy_morison.json).

## Limits

Do not use this implementation as evidence of diffraction/radiation analysis, CFD,
breaking-wave/slamming loads, nonlinear wave theory, dynamic FSI, marine growth,
shielding, fatigue, or code compliance. Coefficients remain engineering inputs and
must be selected and documented for the actual project.
