# Frictionless rigid-plane contact validation

## Scope and convention

This module implements small-displacement, frictionless node-to-rigid-plane
contact. It does **not** yet provide deformable surface-to-surface contact.
For every constraint, `gap = offset + C u`: open contact has `gap > 0`, and
compression has `lambda > 0`. The exact active-set solution enforces
`gap >= 0`, `lambda >= 0`, and `gap*lambda = 0`. The penalty option uses
`lambda = k_p max(-gap, 0)`.

## Independent spring/bar benchmark

A spring of stiffness `k=1000 N/m`, initially `g0=0.01 m` above a rigid
plane, is pressed toward it by `P=20 N`. Exact active contact gives
`u=-g0=-0.01 m`, `lambda=P-k*g0=10 N`, zero penetration, and structural
energy `k*u^2/2=0.05 J`.

For `k_p=100 k`, the independently derived penalty solution is

`penetration=(P-k*g0)/(k+k_p)`, `lambda=k_p*penetration`, and
`contact energy=k_p*penetration^2/2`.

The numerical test compares all three quantities with these expressions and
also compares the penalty reaction with the rigid-contact limit. Its error is
`k/(k+k_p)=0.9901%`, below the project requirement of 3%.

## Hertz helper limitation

`hertz_sphere_on_halfspace` supplies the classical elastic analytical values
`a=(3FR/(4E*))^(1/3)`, `delta=a^2/R`, and `p0=3F/(2*pi*a^2)`. It is a reference
calculator only; its identity test is not claimed as a Hertz finite-element
benchmark. A complete Hertz FE claim requires deformable contact surfaces,
pressure integration, and mesh-convergence evidence, which are out of this
module's present scope.
