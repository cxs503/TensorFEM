# Three-dimensional solid extended validation

The extended HEX8 validation is independent of the original affine patch tests.
Every advertised numerical result is required to have relative error strictly
below 3%.

## Distorted-mesh body-force bar

A prismatic bar with `nu=0` is fixed axially at `x=0`, constrained in the two
transverse directions, and subjected to a uniform axial body force `b`. The
independent one-dimensional continuum solution is

`u(L) = b L^2 / (2 E)`.

Interior nodes are smoothly displaced in all three coordinates while all six
exterior faces remain geometrically exact. This produces non-affine HEX8 maps
and exercises all eight Jacobians. Tests require error reduction from 2 to 6
elements along the bar and a final error below 3%.

## Fixed-free longitudinal bar mode

The same `nu=0` reduction is used to isolate axial motion. A consistent HEX8
mass matrix is integrated with 2x2x2 Gauss quadrature. The independent exact
first angular frequency is

`omega_1 = pi/(2L) sqrt(E/rho)`.

Six elements are used along the axis and the relative frequency error must be
strictly below 3%. The mass test separately checks symmetry, positive
definiteness, and conservation of analytical volume mass in each translational
component.

References: S. Timoshenko, D. H. Young and W. Weaver, *Vibration Problems in
Engineering*, fixed-free uniform rod solution; O. C. Zienkiewicz and R. L.
Taylor, *The Finite Element Method*, isoparametric element and consistent mass
formulations.
