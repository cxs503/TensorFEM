# MacNeal--Harder pinched-cylinder qualification

The full cylinder has length 600, radius 300, thickness 3, `E=3e6` and
`nu=0.3`. Rigid diaphragms support both ends. Two equal diametrically opposed
unit point forces pinch the cylinder at midspan. Symmetry permits an octant
model (`x=0..300`, `theta=0..pi/2`); the corner force in the octant is 0.25.
Displacement is measured radially at the loaded point. Reflection planes
restrain normal translation and the two in-plane rotation-vector components;
the end diaphragm restrains both cross-section translations.

The published radial displacement is `-1.8248e-5`. Source: MacNeal and Harder,
“A proposed standard set of problems to test finite element accuracy,”
*Finite Elements in Analysis and Design* 1 (1985), 3--20,
DOI `10.1016/0168-874X(85)90003-4`.

| octant mesh | displacement | relative error |
|---:|---:|---:|
| 3x3 | -1.798886e-5 | 1.420% |
| 4x4 | -1.772660e-5 | 2.857% |
| 6x6 | -1.726733e-5 | 5.374% |
| 8x8 | -1.763665e-5 | 3.350% |
| 12x12 | -1.806824e-5 | 0.985% |

The reduced/selective integration formulation has a non-monotone coarse-grid
regime: 3x3 and 4x4 happen to lie inside 3%, then 6x6 moves away before the 6/8/12
sequence converges monotonically. Qualification therefore uses 12x12, not the
fortuitous coarse results. The global corotational Newton step is separately run on
3x3; it gives `-1.809425e-5` (0.843% from the published reference) and must
remain below 3%. This is an integration check,
not the fine-mesh qualification result.

Pinched Cylinder and Scordelis--Lo Roof now have sourced responses below 3%.
The curved-shell family remains experimental until a comparable hemispherical
shell benchmark and production sparse nonlinear assembly are available.
