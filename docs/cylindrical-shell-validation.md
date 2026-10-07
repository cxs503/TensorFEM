# Metric-consistent cylindrical Shell4 validation

## Formulation

The element uses the exact cylindrical surface coordinates `(x,s=R theta)`
and a node-wise orthonormal `(axial,tangent,normal)` basis.  In particular it
retains the curvature terms

`epsilon_theta = dv/ds + w/R` and `gamma_theta-z = dw/ds - v/R + beta_theta`.

This distinguishes it from the earlier flat-facet experiment. Membrane strains
are projected at the element centre to avoid curvature-induced membrane
locking, bending uses 2x2 Gauss quadrature, and transverse shear uses selective
one-point integration. The drilling stabilization is difference-only and was
fixed at `1e-8`; it was not fitted to a benchmark response.

Automated tests cover the rigid-translation defect under refinement, an exact
constant biaxial membrane-energy patch, and global load/reaction balance.

## Scordelis--Lo roof

The full roof has length 50, radius 25, half-angle 40 degrees, thickness .25,
`E=4.32e8`, `nu=0`, and downward load 90 per surface area. Diaphragm end
conditions constrain global transverse and vertical translations. The target
is the vertical free-edge displacement `-0.3024`.

| mesh | displacement | relative error |
|---:|---:|---:|
| 2x2 | -9.661737 | 3095.019% |
| 4x4 | -0.216731 | 28.330% |
| 6x6 | -0.285501 | 5.588% |
| 8x8 | -0.296730 | 1.875% |
| 12x12 | -0.301374 | 0.339% |
| 16x16 | -0.302469 | 0.023% |

The coarse reduced-integration meshes are not qualifying results and expose an
hourglass-sensitive pre-asymptotic regime. From 4x4 onward the sequence is
monotonic toward the reference; 8x8 and finer satisfy the strict 3% gate.

Reference: MacNeal and Harder, *Finite Elements in Analysis and Design* 1
(1985), 3--20, DOI 10.1016/0168-874X(85)90003-4.

## Scope

This remains **experimental** rather than a qualified shell capability. The
Scordelis response itself passes at 8x8 and converges to the reference, but the
nodal local-vector interpolation is not exactly objective on a finite angular
patch. Its normalized rigid-translation energy defect decreases from
`9.01e-8` (angle span .2) to `1.53e-9` (.1) and `2.43e-11` (.05), rather than
being machine zero. The 8x8 global vertical equilibrium defect is 0.213%.

Promotion requires a covariant assumed-strain interpolation that preserves
rigid motion exactly while retaining the demonstrated Scordelis convergence.
Pinched cylinder and hemispherical shell remain unclaimed.
