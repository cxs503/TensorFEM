# Plate bending validation

## Scope

`plate.py` implements a differentiable four-node Mindlin-Reissner plate
element with three DOFs per node (`w`, `theta_x`, `theta_y`). Bending uses
2x2 Gauss integration; transverse shear uses one-point selective reduced
integration to avoid thin-plate shear locking. This release does **not** claim
a membrane-coupled shell element.

## Navier simply-supported square benchmark

A square plate of side `a`, thickness `t`, Young's modulus `E`, Poisson ratio
`nu`, and hard simple supports (`w=0`) is loaded by

`q(x,y)=q0 sin(pi x/a) sin(pi y/a)`.

Kirchhoff/Navier thin-plate theory gives the centre deflection

`w0 = q0 a^4/(4 pi^4 D)`, `D=E t^3/[12(1-nu^2)]`.

The corresponding Mindlin result adds transverse-shear deflection
`q0 a^2/(2 pi^2 kappa G t)`. A thick case (`t/a=0.2`) is checked against
that independent closed form.

This is an analytical reference, not a same-code comparison. Automated tests
require the 16x16 result to have relative error strictly below 3% for both
`t/a=1e-2` and the locking-sensitive `t/a=1e-3`. The latter is also checked
on 4x4, 8x8 and 16x16 meshes for monotone convergence.

## Recorded convergence

| t/a | mesh | centre deflection | reference | relative error |
|---:|---:|---:|---:|---:|
| 0.01 | 4x4 | 2.701676e-3 | 2.802613e-3 | 3.6015% |
| 0.01 | 8x8 | 2.779598e-3 | 2.802613e-3 | 0.8212% |
| 0.01 | 16x16 | 2.798106e-3 | 2.802613e-3 | 0.1608% |
| 0.001 | 4x4 | 2.699934 | 2.802613 | 3.6637% |
| 0.001 | 8x8 | 2.777992 | 2.802613 | 0.8785% |
| 0.001 | 16x16 | 2.796531 | 2.802613 | 0.2170% |
| 0.2 | 16x16 | 4.291057e-7 | 4.293572e-7 | 0.0586% |

The coarse grids are convergence evidence, not qualifying advertised results;
the formal 16x16 results are all strictly below 3%. The t/a=0.001 sequence
retains the thin-limit deflection and therefore detects shear locking.

Reference: S. Timoshenko and S. Woinowsky-Krieger, *Theory of Plates and
Shells*, 2nd ed., McGraw-Hill, 1959, Navier solution for simply-supported
rectangular plates.
