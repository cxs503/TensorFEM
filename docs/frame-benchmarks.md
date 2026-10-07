# 2-D frame benchmark contract

The frame solver uses Euler--Bernoulli theory, small displacement, linear
elastic material, and three degrees of freedom per node (`ux`, `uy`, `rz`).

Formal acceptance cases are the cantilever under a tip force and the same
beam under a full-span uniform transverse load. Their reference quantities
are respectively `PL^3/(3EI)`, `PL^2/(2EI)`, `qL^4/(8EI)`, and
`qL^3/(6EI)`. Tests require every primary quantity to have relative error
below 3%. Since the exact consistent load vector and cubic beam field are
used, these cases are exact to floating-point precision with one element.

The rotated-member case additionally verifies coordinate transformation,
axial deformation `PL/(EA)`, and global force equilibrium. A differentiable
test verifies that response gradients propagate to section inertia.
