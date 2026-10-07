# Deformable axisymmetric Hertz contact

## Scope and formulation

`hertz_axisymmetric.py` is a small, auditable deformable-contact solver. The
elastic half-space surrogate uses axisymmetric Q4 elements with radial,
vertical, hoop and shear strains and exact `2*pi*r` Gauss integration. A rigid
parabolic sphere imposes frictionless unilateral contact through a nodal
penalty active set on the top surface. The nonlinear solve couples elastic
deformation and the computed contact pressure; pressure is not prescribed.

The domain is a graded finite cylinder, 24 contact radii wide and deep. The
axis and bottom symmetry/support conditions and the remote radial constraint
truncate the half-space. This is an **axisymmetric small-strain Hertz
verification**, not a general three-dimensional surface-contact solve.

## Independent reference

For indentation `delta=0.025`, sphere radius `R=10`, half-space
`E=100000`, `nu=0.3`, the rigid-sphere Hertz oracle is

- `E*=E/(1-nu^2)`;
- `a=sqrt(R*delta)=0.5`;
- `F=4 E* a^3/(3R)=1831.5018315018315`;
- `p0=3F/(2*pi*a^2)=3497.910837184513`.

Contact radius and peak pressure are recovered from the computed active nodal
pressure field using the Hertz identity `p^2=p0^2-(p0^2/a^2)r^2`. This fit is
post-processing only: it does not enter equilibrium or prescribe the field.
Raw active extent and raw nodal peak remain in the result for auditing.

## Mesh convergence and qualification

| radial x depth elements | load | load error | radius error | peak-pressure error | overlap/delta |
|---:|---:|---:|---:|---:|---:|
| 16 x 16 | 1720.7875 | 6.0450% | 14.4294% | 8.3889% | 3.0306% |
| 24 x 24 | 1931.0021 | 5.4327% | 4.1924% | 5.6309% | 2.9477% |
| 48 x 48 | 1882.8952 | 2.8061% | 2.7738% | 2.2767% | 2.8017% |

The first two meshes are convergence evidence and fail the 3% capability
gate. The 48 x 48 result is the qualifying discretisation: load, radius,
peak pressure, full pressure-distribution weighted L2 error and normalized
penalty overlap are each strictly below 3%. Errors in all four nontrivial
Hertz outputs decrease along the stated 16/24/48 sequence. Indenter travel is
the prescribed displacement and is checked independently against the oracle;
the normalized contact overlap is reported separately rather than hidden in
that quantity.

A separate truncation study holds contact-zone resolution approximately
constant while increasing the cylinder from 24 to 32 contact radii in width
and depth (40x40 to 53x53 elements). Load, fitted contact radius and fitted
peak pressure change by less than 3%. The reaction recovered from the remote
supports also balances the independently integrated contact resultant to
relative tolerance `1e-10`.

The penalty is fixed in physical units across the refinement sequence; it is
not retuned per mesh. Nonnegative pressure, solver convergence, stiffness
symmetry and lack of materially negative elastic eigenvalues are automated
contracts. The finite cylinder, nodal pressure and profile recovery remain
limitations; this module does not qualify arbitrary 3-D Hertz geometry,
frictional Hertz contact or nonmatching mortar discretisation.

```bash
PYTHONWARNINGS=error PYTHONPATH=src python -m pytest -q \
  tests/test_hertz_axisymmetric.py
```
