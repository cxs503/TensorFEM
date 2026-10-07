# TET4 small-strain J2 plasticity validation

## Scope

`solid_plasticity.py` integrates the existing backward-Euler J2 radial return
at the single integration point of a constant-strain TET4.  The global solver
uses load control and Newton iterations.  Each iteration starts from the last
committed integration-point state; state is committed only after convergence,
so a failed increment rolls back automatically.  This is a small-strain model,
not finite-strain plasticity.

The algorithmic tangent is differentiated from the return-map implementation
itself and is checked against a central difference of the assembled internal
force.

## Verification contract

A unit right tetrahedron is constrained so its three normal strain components
are free while rigid modes and shears are suppressed. A uniform uniaxial
generalized nodal load `Q=sigma*V` produces a homogeneous uniaxial-stress
solution (by virtual work for the affine mode). Material data are
`E=200000`, `nu=0.3`, initial yield stress `250`, and linear isotropic
hardening modulus `H=10000`.

At final stress `sigma=400`, the exact monotonic strain is

`epsilon = sigma/E + (sigma-sigma_y)/H = 0.017`.

The automated test requires stress and displacement errors below 3% (the
recorded errors are at floating-point roundoff), and convergence in no more
than three Newton iterations.  Complete unloading checks the residual plastic
strain `(400-250)/10000 = 0.015`, zero final stress, and the exact closed-path
work `V*(sigma_max^2-sigma_y^2)/(2H)`.  A forced failed step
checks that committed state remains byte-for-byte unchanged.

Reference: J. C. Simo and T. J. R. Hughes, *Computational Inelasticity*,
Springer, 1998, Chapters 1 and 3 (associative J2 plasticity and return mapping).
