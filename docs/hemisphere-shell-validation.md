# Hemispherical shell with an 18-degree hole

This is the MacNeal--Harder doubly curved shell benchmark. The midsurface is a
hemisphere of radius 10 with a circular hole extending to polar angle 18
degrees. Thickness is 0.04, `E=6.825e7`, and `nu=0.3`. A quarter model spans
azimuth 0 to 90 degrees. Unit alternating radial point loads act at its two
equator corners. On each meridional symmetry plane, normal translation and the
two in-plane rotation-vector components are restrained. One vertical DOF is a
pure rigid-translation gauge; its reaction is required to remain roundoff.

The published radial displacement at the loaded equator point is `0.0924`.
Source: MacNeal and Harder, “A proposed standard set of problems to test finite
element accuracy,” *Finite Elements in Analysis and Design* 1 (1985), 3--20,
DOI `10.1016/0168-874X(85)90003-4`.

The latitude--longitude corner facets are planar to roundoff, while adjacent
facet normals vary in both surface directions. `projected_shell4_stiffness`
uses diagonal centre tangents and a centre-normal projection. Membrane natural shear and
plate transverse shear retain the existing selective-integration treatment.

| quarter mesh | displacement | relative error |
|---:|---:|---:|
| 4x4 | 0.061188 | 33.779% |
| 6x6 | 0.075059 | 18.767% |
| 8x8 | 0.085976 | 6.952% |
| 12x12 | 0.091763 | 0.690% |
| 16x16 | 0.093203 | 0.869% |
| 20x20 | 0.093697 | 1.404% |

Displacement converges monotonically through the qualification mesh; 12x12
and finer are below 3%. The 12x12 result also stays below 3% for drilling
factors `1e-7`, `1e-6`, and `1e-5`, so qualification is not based on tuning a
single penalty to the answer. Free-DOF residual and the vertical gauge reaction
are independently required at roundoff.

Together with sourced Scordelis--Lo and Pinched Cylinder results, this closes
the three classical response cases. The projected linear spherical element is
still separate from the cylindrical finite-rotation nonlinear formulation;
general nonlinear doubly curved shells remain experimental.
