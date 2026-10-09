# Public snap-through shell qualification audit

## Candidate reviewed

The reviewed public case is the clamped shallow spherical shell under a static
ring load in E. E. Karatas and R. F. Yuksel, “Snap-through Buckling of Shallow
Spherical Shells under Ring Loads,” *Teknik Dergi* 31 (2020), DOI
`10.18400/tekderg.565095`. The accessible article explicitly gives the
experimental-comparison configuration `R=254 mm`, `t=0.3810 mm`, depth
parameter `eta=0.0618`, and ring diameter `d=25.4 mm`; it defines a clamped
circular edge and axisymmetric apex conditions. It also states that this case
has limit points.

## Why it is not qualified

The published limit coordinates are presented as plotted load--deflection
curves, not a numerical table. Digitizing those pixels and treating the result
as a `<3%` reference would not be an independently auditable value. The
material constants associated with the plotted experimental comparison also
cannot be recovered reliably from the machine-readable article text.

More importantly, the current `general_shell_arc_problem` accepts a fixed
reference load vector. A ring load attached to the deforming shell requires a
configuration-dependent residual and its consistent follower-load tangent.
Running the existing fixed-direction load would solve a different problem.

`snapthrough_shell_qualification` records these blockers in JSON and
`require_qualified()` fails closed. No critical value is stored, inferred or
registered. The existing Crisfield solver, rollback/restart and general-shell
adapter remain verified independently, but they are not used to claim this
published shell case.

## Promotion requirements

1. Obtain tabulated author/reference limit coordinates or a primary-source
   numerical dataset, including material constants.
2. Add surface/ring follower-load residual and exact load-stiffness tangent.
3. Reproduce ascending, descending and post-snap branches on at least three
   meshes and two arc sizes.
4. Require fine-mesh peak load and displacement errors strictly below 3%, plus
   spatial and continuation-step changes below 3%.

Until all four conditions pass, the platform truthfully reports the public
snap-through shell as blocked rather than substituting the von Mises arch or an
invented shell reference.
