# Rotation-controlled large-rotation shell pure bending

This is the quarter-circle member of the classical large-displacement
cantilever pure-bending benchmark used for geometrically nonlinear beam and
shell formulations. The strip is length 10, width 1, thickness 0.1,
`E=1.2e6`, and `nu=0`. The root is clamped and cross-section rotation varies
linearly to 90 degrees at the free end. Rotation control is used because it is
well-conditioned at large angle and directly audits the full equilibrium path.

Pure bending has an independent circular-elastica solution. For angle
`theta`, curvature is `theta/L`, radius is `L/theta`, tip coordinates are
`x=R sin(theta)`, `z=R(1-cos(theta))`, and reaction moment is
`M=E I theta/L`, with `I=b t^3/12`. At 90 degrees the exact tip is
`(20/pi,0,20/pi)=(6.366198,0,6.366198)` and moment is `15.707963`.

| mesh | tip x=z | vector relative error | moment error |
|---:|---:|---:|---:|
| 1x1 | 7.071068 | 11.072% | roundoff |
| 2x1 | 6.532815 | 2.617% | roundoff |
| 4x1 | 6.407289 | 0.645% | below 3e-7 |

The coarse-to-fine sequence is monotonic and the 4x1 result is below 3%.
Intermediate accepted rotations are checked against the same analytical
circular load-displacement path, not only the final point. The full 90-degree
rotation makes this a strong geometric-nonlinearity test; material behavior
remains linear elastic.

Related literature: K.-J. Bathe and S. Bolourchi, “Large displacement analysis
of three-dimensional beam structures,” *International Journal for Numerical
Methods in Engineering* 14 (1979), 961--986. The numerical reference used here
is the explicitly stated circular-elastica formula, not a digitized graph.
