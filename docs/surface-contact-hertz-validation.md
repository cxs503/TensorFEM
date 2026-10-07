# Low-order surface contact and Hertz validation

## Capability boundary

`surface_contact3d.py` implements node-quadrature surface-to-surface contact.
TRI3/QUAD4 slave facets supply current tributary areas; every active slave
node searches all current master facets, evaluates the three-dimensional
penalty/augmented-Lagrangian and Coulomb law, and distributes the opposite
reaction to master nodes using barycentric coordinates.

This low-order formulation is not mortar contact and does not pass a
continuous-pressure patch test on arbitrary nonmatching meshes. It currently
has no broad-phase acceleration, smoothing across master facet normals, or
automatic self-contact topology construction.

## Surface-contact verification

- A unit QUAD4 patch penetrating `0.002` with pressure penalty `200000`
  produces resultant force `400`; assembled slave and master resultants are
  equal and opposite.
- Uniform tangential motion verifies the Coulomb limit and partitions work
  into recoverable tangential energy plus positive friction dissipation.
- Translating the complete slave patch across a master seam activates the
  adjacent face while preserving the analytical normal resultant.
- Maximum penetration is explicitly reported. Degenerate facets and a state
  whose topology differs from the current slave mesh fail closed.

## Hertz benchmark and negative evidence

The qualified Hertz test is deliberately a **traction-integration benchmark**,
not a solved deformable-contact benchmark. For a sphere of radius `R` pressed
against an elastic half-space by force `F`, the independent classical oracle is

`a=(3 F R / (4 E*))^(1/3)`,
`p(r)=p0 sqrt(1-r^2/a^2)`, and `p0=3F/(2 pi a^2)`.

Midpoint annular quadrature integrates that pressure. For `F=1000`, `R=0.05`,
two identical materials with `E=210e9`, `nu=0.3`, the reference radius is
`6.875344335370708e-4`. The 64-ring integrated force is
`1000.532395418`, a relative error of `0.05324%`, below the 3% gate. Errors
decrease monotonically over 8, 16, 32 and 64 rings.

A rigid sphere on independent penalty springs is retained as negative
evidence. Its load scales as indentation squared, while Hertz load scales as
indentation to the power `3/2`. Even after calibration at one indentation it
has 100% error at four times that indentation. Therefore TensorFEM does **not**
claim that the current penalty surface discretisation solves Hertz elastic
half-space contact. Such a claim requires deformable solid meshes, converged
contact-pressure fields and mesh refinement against the Hertz solution.

Run with warnings promoted to errors:

```bash
PYTHONWARNINGS=error PYTHONPATH=src python -m pytest -q \
  tests/test_surface_contact3d.py
```
