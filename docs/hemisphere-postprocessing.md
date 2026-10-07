# Hemisphere industrial post-processing closure

The dependency-free post-processing path consumes the existing
`HemisphereResult`; it never resolves or modifies the benchmark.  It maps the
solver layout `[ux, uy, uz, rx, ry, rz]` to nodal translation and rotation,
computes deformed coordinates, retains only restrained-DOF reactions, and
reconstructs the two published quarter-model point loads.

Outputs include:

- legacy ASCII VTK with undeformed mesh, displacement, deformed coordinates,
  reaction force/moment, and displacement magnitude;
- versioned JSON with all fields, units, mesh counts, loaded-node probe,
  meridional path, extrema, reference value, errors, balances and pass/fail;
- a concise Markdown validation report.

Units are explicit metadata rather than an implicit conversion.  The original
benchmark uses a consistent unit system; callers may label it (the tests use
inch and pound-force), but values are not rescaled.

The 16x16 qualification mesh is checked against the published displacement
`0.0924` with a three-percent response limit.  Translational equilibrium and
free residual use `1e-7` normalized/absolute gates.  Moment equilibrium is
normalized by radius 10 and total absolute applied force, with a three-percent
spatial-discretization gate.  The latter is intentionally reported separately:
the projected faceted shell approaches rotational balance with refinement and
must not be mistaken for roundoff-level conservation.

Tests verify the coordinate and six-DOF mapping, exact probe index, path node
order, force and moment balance, reference error, JSON tensor round-trip, VTK
geometry round-trip, metadata, fields, and fail-closed schema handling.

```bash
PYTHONWARNINGS=error pytest -q tests/test_hemisphere_postprocess.py
python examples/hemisphere_postprocess.py
```
