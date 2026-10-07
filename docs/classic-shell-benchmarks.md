# Classical curved-shell benchmarks

## Status: experimental, not qualified

The current `shell4_stiffness` is a flat-facet Mindlin shell. Adjacent facets
have different local frames and therefore the assembled mesh represents folded
curvature rather than projecting a curved roof onto a plane. This is sufficient
to run classical curved-shell problems, but it is not sufficient to pass them.

## Scordelis--Lo roof

The full roof uses length 50, radius 25, half-angle 40 degrees, thickness 0.25,
Young's modulus 4.32e8, Poisson ratio zero and a downward surface load of 90.
Both curved ends are supported by rigid diaphragms. The comparison quantity is
the vertical displacement at the midspan of a longitudinal free edge; the
widely reported reference is -0.3024.

| axial x circumferential mesh | computed displacement | relative error |
|---:|---:|---:|
| 2 x 2 | -2.480040 | 720.119% |
| 4 x 4 | -0.492837 | 62.975% |
| 6 x 6 | -0.401343 | 32.719% |
| 8 x 8 | -0.405557 | 34.113% |
| 12 x 12 | -0.425768 | 40.796% |
| 16 x 16 | -0.437542 | 44.690% |
| 20 x 20 | -0.444002 | 46.826% |

The sequence does not converge to the reference. A drilling-factor sensitivity
study also showed a material change in the response; choosing a larger penalty
can make one mesh appear close to the reference, but that would be calibration
to the answer rather than verification. Consequently this case is deliberately
excluded from the qualifying benchmark registry and the automated test asserts
that it remains unqualified. Pinched-cylinder and hemispherical-shell cases are
not claimed because they exercise the same unresolved formulation weakness.

The implementation, geometry, consistent facet loading and reaction equilibrium
remain as an executable experimental regression for development of a proper
curved shell (MITC4/assumed-strain treatment with objective drilling control).

## Traceable references

- MacNeal, R. H. and Harder, R. L., “A proposed standard set of problems to
  test finite element accuracy,” *Finite Elements in Analysis and Design* 1
  (1985), 3–20, <https://doi.org/10.1016/0168-874X(85)90003-4>.
- Scordelis, A. C. and Lo, K. S., “Computer analysis of cylindrical shells,”
  *Journal of the American Concrete Institute* 61 (1964), 539–562.

## Acceptance criterion

Promotion requires monotone or demonstrably asymptotic mesh convergence and a
strict relative error below 3% without tuning stabilization against the target.
