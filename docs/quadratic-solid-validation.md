# TET10 quadratic solid validation

## Contract

`quadratic_solid.py` supplies a conforming ten-node tetrahedron with the node
order `(0,1,2,3,01,12,20,03,13,23)`. Geometry and displacement use the same
quadratic interpolation. Element stiffness uses the five-point degree-three
Keast rule; every integration point is rejected if its Jacobian is non-positive
or numerically degenerate. Shared mesh edges produce one shared midside node.

## Verification evidence

* Kronecker delta, partition of unity and zero derivative-sum identities pass
  at machine precision.
* A complete quadratic displacement field is reproduced at an interior point;
  all six strain components agree with the analytical derivatives within
  `2e-14`.
* A constant-strain patch has an energy error below `2e-13`, including the
  public global assembly and prescribed-displacement solve path.
* A deliberately curved/inverted edge triggers the integration-point Jacobian
  guard.

## Formal bending benchmark

A `10 x 1 x 1` brick partition is split into tetrahedra and upgraded to a
conforming TET10 mesh. Dimensions are `L=10`, `b=h=1`, `E=1e7`, `nu=0.3`.
The root is clamped and unit uniform end shear is integrated consistently on
the quadratic triangular faces (zero corner and `A/3` midside weights).

The reference is the Timoshenko cantilever displacement

`delta = P L^3/(3 E I) + P L/(k G A)`, with `I=bh^3/12`, `G=E/[2(1+nu)]`,
and rectangular-section `k=5/6`. The computed centre-tip displacement is
`3.932990e-4`; reference is `4.03120e-4`, giving `2.4362%` relative error.
This is strictly below the project's 3% qualification threshold.

The matching TET4 mesh computes `8.84341e-5`, an error of `78.063%`. Thus the
formal TET10 result is over 32 times closer to the reference on identical
corner-node topology. Coarser TET10 meshes (`nx=2,4,6,8`) give errors of
`12.543%, 5.665%, 3.757%, 2.907%`, providing an explicit convergence sequence;
only qualifying meshes below 3% are advertised as formal results.

References: S. P. Timoshenko, *Strength of Materials*, beam deflection; O. C.
Zienkiewicz, R. L. Taylor and J. Z. Zhu, *The Finite Element Method*, quadratic
isoparametric tetrahedra and tetrahedral quadrature.
