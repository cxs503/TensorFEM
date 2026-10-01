# Three-dimensional solid edge-case validation

All advertised benchmark errors are strictly below 3%. Double precision is
used. The tests are automated in `test_solid3d_edgecases.py`.

## Free-body modes

A unit rectangular body discretised independently by six TET4 elements and by
one HEX8 element is left unconstrained. The generalized eigenproblem must have
exactly six rigid translations/rotations. Their maximum absolute eigenvalue,
normalised by the largest elastic eigenvalue, is required below `1e-10`; the
seventh mode must be strictly elastic.

## TET4 consistent mass and axial dynamics

The exact TET4 consistent scalar mass is
`rho V / 20 * [[2,1,1,1], ...]`, copied to each translational component. Its
assembled mass is checked against `rho * volume` in every direction.

A fixed-free prismatic rod is meshed with TET4 elements. Transverse degrees of
freedom are constrained and Poisson ratio is zero, isolating the longitudinal
wave. Its first frequency is compared with
`omega_1 = pi/(2L) sqrt(E/rho)` and must differ by less than 3%.

The same assembled TET4 stiffness and consistent mass are used in an
average-acceleration Newmark integration. Starting in the first FE eigenmode
with zero velocity, modal displacement is compared over three periods with
`cos(omega_h t)`; maximum absolute error must be below 3%.

## Near-incompressible HEX8 diagnostic (experimental)

The fully integrated displacement-based HEX8 is known to exhibit volumetric
and bending locking near `nu=0.5`. A slender cantilever at `nu=0.4999` is
therefore retained as a **negative diagnostic**: the test requires the error
to exceed 3%, proving that the unsupported regime is detected. This case is
not an advertised passing benchmark and must not be interpreted as validated
near-incompressible capability. A mixed `u-p`, selective-reduced-integration,
or verified B-bar formulation is required before promotion.

References: O. C. Zienkiewicz, R. L. Taylor and J. Z. Zhu, *The Finite Element
Method: Its Basis and Fundamentals*, 7th ed.; T. J. R. Hughes, *The Finite
Element Method: Linear Static and Dynamic Finite Element Analysis*.
