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

## Corrected coordinate-gradient qualification

The Mindlin gradient transformation on skew facets now uses the inverse
transpose for a Jacobian with natural-coordinate rows. The previous values
are superseded: they contained spurious rigid-rotation stiffness.

| quarter mesh | displacement | relative error |
|---:|---:|---:|
| 4x4 | 0.1015827104 | 9.937998% |
| 6x6 | 0.0987380625 | 6.859375% |
| 8x8 | 0.0972050000 | 5.200217% |
| 12x12 | 0.0958018428 | 3.681648% |
| 16x16 | 0.0952090065 | 3.040050% |
| 20x20 | 0.0948885397 | 2.693225% |
| 24x24 | 0.0946845941 | 2.472504% |
| 28x28 | 0.0945409711 | 2.317068% |
| 32x32 | 0.0944331456 | 2.200374% |

The default fixed drilling factor remains 1e-6. The loaded-point displacement
converges; 12x12 and 16x16 fail the 3% gate after correction. At 24x24 the
1e-7 factor still gives 4.0051% error, so the old coarse robustness claim is
withdrawn. The complete report checks the same two-decade bracket at 40x40,
retaining the failed coarse bracket. Force and moment equilibrium must each
be below 1e-7, independent of the displacement comparison.

This is response qualification, not independent shell stress qualification.
General nonlinear doubly curved shells remain experimental.

[Full report and raw evidence](benchmarks/tutorials/hemisphere_hole_full.md).
