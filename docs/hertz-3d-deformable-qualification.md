# Three-dimensional two-deformable-body Hertz qualification

## Current status: blocked

The classical two-material oracle is implemented, but TensorFEM does not yet
claim a qualified general 3-D Hertz solve. `solid3d.py` provides TET4/HEX8
linear elasticity and `global_contact3d.py` provides transactional
node-to-TRI3 contact. They are useful ingredients, but their current public
path does not construct two curved solid bodies, integrate a continuous
surface pressure, or provide the required contact-zone and remote-domain
studies. The qualified axisymmetric rigid-indenter case is separate evidence.

`hertz_3d_qualification.py` is a fail-closed evidence gate for the missing
capability. A result can be marked qualified only when all of the following
are present:

- sphere and half-space are both discretised by three-dimensional solids;
- contact pressure is solved and integrated, not sampled from the oracle;
- at least three strictly refined contact-zone meshes show monotone convergence;
- integrated load, support reaction, contact radius, indentation, peak
  pressure, and pressure-field L2 error satisfy the strict `<3%` gate;
- a larger remote domain, at retained contact resolution, changes all primary
  outputs by less than 3%.

The gate deliberately rejects rigid spheres, axisymmetric runs, analytical
pressure integration, missing domain studies, and values equal to the 3%
threshold. This prevents existing partial capabilities from being relabelled
as an industrial 3-D validation.

## Required solver work

The next implementation must couple assembled sphere and half-space solid
stiffnesses into the same nonlinear residual, add curved surface-to-surface
quadrature (or mortar contact), recover pressure at quadrature points, and
support symmetry and remote-boundary constraints. The benchmark parameters
remain the externally contracted `F=1000 N`, `R=0.01 m`, and identical
`E=210 GPa`, `nu=0.3` materials.

```bash
PYTHONWARNINGS='error,ignore:Failed to initialize NumPy' PYTHONPATH=src \
  python -m pytest -q tests/test_hertz_3d_qualification.py
```
