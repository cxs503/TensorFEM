# Shell4 prestress eigenbuckling qualification

This qualification uses the production flat-facet `shell4_stiffness` elastic
matrix and a newly integrated Q4 initial-stress matrix. It is not a Navier or
single-mode wrapper: every mesh assembles the 24-DOF Shell4 elements, applies
simply-supported transverse boundary conditions, statically condenses the
Mindlin rotations, and solves `K phi = lambda Kg phi`.

The auditable reference is the classical Navier solution for a simply
supported isotropic rectangular plate under uniform uniaxial compression,

`Nx_cr = D [(m pi/a)^2 + (n pi/b)^2]^2 / (m pi/a)^2`,

where `D = E t^3/[12(1-nu^2)]`. The qualification searches integer wave
numbers rather than assuming their value. A unit square with `E=210 GPa`,
`nu=0.3`, and `t=0.01 m` is evaluated on 4x4, 8x8, and 16x16 meshes. The fine
mesh must differ from Navier by less than 3%, the last mesh change must be less
than 3%, and the sequence must converge monotonically. A second test verifies
that equal biaxial compression halves the square-plate critical load.

Run the evidence with:

```bash
PYTHONPATH=src python -m pytest -q tests/test_shell4_buckling_qualification.py
PYTHONPATH=src python examples/marine/shell4_buckling.py
```

## Capability boundary

This evidence qualifies linear initial-stress eigenbuckling of flat isotropic
Shell4 plates with constant element stress resultants. It does not yet qualify
stress recovery from a nonlinear prebuckling solve, curved-shell geometric
stiffness, mode switching, imperfections, material plasticity, or nonlinear
postbuckling. The existing `general_shell_arc_problem` exposes the production
general-shell residual and consistent tangent to Crisfield continuation, but
its published qualification remains an algorithm-level von Mises arch case.
No plate/shell postbuckling claim is made here until a full discrete shell path
has spatial and continuation-step convergence against a public numerical
reference below the 3% gate.

Reference: S. Timoshenko and J. Gere, *Theory of Elastic Stability*, 2nd ed.,
McGraw-Hill, 1961, chapter 9 (simply supported rectangular plates).
