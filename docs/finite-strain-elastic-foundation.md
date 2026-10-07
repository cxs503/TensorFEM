# Finite-strain three-dimensional elastic foundation

This module is a Total-Lagrangian TET4 formulation with `F = I + Grad(u)` and
compressible Neo-Hookean energy. Internal force uses first Piola stress in the
reference configuration. The exact material-plus-geometric tangent is obtained
by differentiating the element residual, and adaptive global Newton steps are
restartable. Inverted or singular deformation gradients fail closed.

Qualification includes a 1.37-radian superposed rigid rotation objectivity
test, a zero-energy pure rotation test, tangent finite differences, and a
twelve-element bar at 50% extension. The latter is checked against the
independent closed-form nominal stress

`P11 = mu*(lambda - 1/lambda) + lame_lambda*log(lambda)/lambda`.

## Deliberate boundary

This is finite-strain **elasticity**, not finite-strain plasticity. Existing
small-strain J2 history is not reused or relabelled. A defensible plastic
extension still requires a multiplicative split `F=Fe Fp`, isochoric evolution
of `Fp`, an objective stress/yield measure, exponential-map or equivalent
integration, hardening state, plastic dissipation checks, and a consistently
linearized algorithmic tangent. Those remain quantified implementation gaps.
