# Ship and offshore structural screening models

This capability package introduces two deliberately bounded, traceable models
for preliminary ship-structure calculations.  All quantities use a consistent
SI unit system in the supplied JSON cases.  These models are verification aids;
they are **not classification-society certified**, and do not replace rule
scantlings, fatigue, ultimate-strength, hydroelasticity, or local-detail checks.

## Hull-girder longitudinal bending

`solve_hull_girder_uniform_load` idealizes the hull as a simply supported
Euler--Bernoulli beam.  The still-water and wave line-load components are kept
separate in the input and superposed for a linear load case.  Consistent beam
loads are assembled through the standard Frame2D solver.  Midship deflection
and moment are independently checked against

`w_mid = 5 q L^4/(384 E I)` and `M_mid = q L^2/8`.

This is an equivalent global bending model.  It does not model buoyancy/load
curve balance, shear lag, torsion, whipping, springing, yielding, or buckling.

## Longitudinally stiffened panel

`stiffened_panel_sine_benchmark` smears identical longitudinal stiffeners into
the orthotropic rigidity `D_x = D_plate + E_s A_s z_s^2/s`.  A simply supported
panel under one sinusoidal pressure mode is solved by numerical Rayleigh--Ritz
quadrature and checked against the closed-form Navier amplitude

`w0 = q0/[D_x alpha^4 + 2 D_xy alpha^2 beta^2 + D_y beta^4]`.

The model verifies equivalent elastic bending only.  It excludes discrete
stiffener torsion, plate/stiffener local buckling, initial imperfections,
plastic collapse, weld hot spots, and fatigue.

Both automated benchmarks enforce relative error below 3%. Run them with:

```bash
PYTHONWARNINGS=error PYTHONPATH=src python -m pytest -q tests/test_marine_structures.py
PYTHONPATH=src python examples/run_marine_structures.py
```
