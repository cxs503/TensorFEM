# Experimental curved-shell consistent tangent and Roof audit

`shell_consistent` parameterizes each nodal orientation with an exponential
map. Its internal force is the exact derivative of the corotational energy,
and its 24x24 tangent is the derivative of that same force. This includes the
derivatives of the proper-Kabsch frame and nodal rotations; it is not a frozen
or separately fitted stiffness.

A directional central difference at a finite translation/rotation state must
match the tangent product within `2e-6`, while Hessian symmetry must be within
`2e-10`. At the reference state, the three rigid translations are null modes
within `2e-12` after stiffness normalization. Tests execute with warnings as
errors.

## Scordelis--Lo Roof

The recognized MacNeal--Harder roof remains the auditable response driver:
length 50, radius 25, half-angle 40 degrees, thickness 0.25, `E=4.32e8`,
`nu=0`, vertical surface load 90, and diaphragm end conditions. The published
free-edge midpoint displacement is `-0.3024`.

| mesh | displacement | relative error |
|---:|---:|---:|
| 6x6 | -0.285501 | 5.588% |
| 8x8 | -0.296730 | 1.875% |
| 12x12 | -0.301374 | 0.339% |

Reference: MacNeal and Harder, *Finite Elements in Analysis and Design* 1
(1985), 3--20, DOI `10.1016/0168-874X(85)90003-4`.

The 8x8 and 12x12 Roof responses pass the strict 3% gate, and the new element
kinematics/tangent pass their independent objectivity and consistency gates.
The overall curved-shell feature remains **experimental**: the nonlinear
global assembly/step is not yet integrated and Pinched Cylinder plus
Hemispherical Shell have not been qualified. No reference values are inferred
for those absent benchmarks.
