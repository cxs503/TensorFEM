# Large-angle stability (`GZ` curve) of a wall-sided box barge

`tensorfem.marine_large_angle_gz` closes the capability gap that
`tensorfem.marine_hydrostatics` documents as out of scope: **a large-angle `GZ`
curve and cross-curves**.  Hydrostatics stops at |heel| <= 10 deg with the
metacentric relation `GZ = GM sin(phi)`; this module evaluates the righting arm
from exact section geometry for heel angles up to 90 deg, with no mesh and no
quadrature.

## Method

For a prismatic barge of beam `B`, moulded depth `D`, draft `T`, symmetric
loading (longitudinal centre of gravity on the section centreline, no trim) the
heeled displacement is a two-dimensional problem.  In body axes (x toward
starboard, z up from the keel) a heel `phi` about the longitudinal axis puts the
waterline at

```
-x sin(phi) + z cos(phi) = d,      n = (-sin phi, cos phi)
```

so the immersed part of the section rectangle `[-B/2, B/2] x [0, D]` is
`{r : n . r <= d}`.  Constant displacement (`V = L B T` at every heel, exact for
a symmetric prismatic hull) fixes `d` through `Area(d) = B T`; the monotone
bracketed root is refined by bisection to machine precision (<= 200 steps,
bracket from the corner extrema of `n . r`).  Area and centroid of the clipped
convex polygon come from the shoelace / section formula, and the righting arm is

```
GZ(phi) = xB cos(phi) + (zB - KG) sin(phi),
KN(phi) = xB cos(phi) + zB sin(phi),   GZ = KN - KG sin(phi).
```

One geometric sweep therefore produces a whole cross-curve family, which is the
standard naval-architecture decomposition.

## Verification basis

The implementation is scored against *closed forms*, not against itself.
Reference barge `B = 12 m, D = 6 m, T = 2 m` (`KG = 2.1 m`):

| quantity | value |
| --- | --- |
| `KB`, `BM`, `KM`, `GM` | 1.0000, 6.0000, 7.0000, 4.9000 m |
| wall-sided band end (min of bilge / deck) | 18.43495 deg |
| max `|GZ_geom - GZ_wall_sided|` inside the band | 1.332e-15 m |
| corresponding relative error (max `|GZ|` in band 1.612 m) | 8.26e-16 |
| max `|GZ_geom - GZ_wall_sided|` outside the band | 800128136434071378374934856204288.0000 m (29907880522234843295157916832628736.0%) |
| `GZ(90 deg)` vs exact `D/2 - KG` | 0.9000000000 vs 0.9000000000 m |
| max `GZ` and its angle | 2.67531 m at 43.00 deg |
| dynamic stability area `int GZ dphi` (0-90 deg) | 2.899948 m*rad |

Anchors used by `tests/test_marine_large_angle_gz.py` (39 tests):

1. **Wall-sided formula** `GZ = sin(phi) (GM + BM tan^2(phi)/2)` (Biran,
   *Ship Hydrostatics and Stability*, 2003, Sec. 4.9; Lewis, *Principles of Naval
   Architecture* I, 1988, Sec. 2.6) reproduced to `1e-12` relative, together
   with its geometric consequences `xB = BM tan(phi)` and
   `zB = KB + BM tan^2(phi)/2`.
2. **Sharp validity boundary**: the closed form is exact up to the bilge/deck
   limit and demonstrably wrong past it, with the residual growing monotonically
   (`> 10%` at band end `+10 deg`).
3. **Exact 90 deg geometry**: `xB = B/2 - B T/(2 D)`, `zB = D/2`,
   `GZ = D/2 - KG`.
4. **Metacentric slope**: `GZ/sin(phi) -> GM` and a second-order Richardson
   extrapolation of the finite difference returns `GM` to `1e-6` relative,
   cross-checked against `marine_hydrostatics.analyze_box_barge`.
5. **Constant displacement**: immersed area equals `B T` at every heel to
   `1e-12` relative.
6. **Independent second formulation**: a closed-form antiderivative of the
   clamped waterline height, integrated on a 200 001-point grid, reproduces
   `h`, area, `xB`, `zB` in all three regimes to `1e-8` relative.
7. **`KN` decomposition**: `GZ(KG) = KN - KG sin(phi)` for several `KG`, `KN(0) = 0`,
   `dKN/dphi|0 = KB + BM`, `KN(90 deg) = D/2`.
8. **Dynamic stability**: `int_0^phi GZ dphi = GM(1-cos) + BM(sec + cos - 2)/2`
   matched to `1e-6` and shown to converge at second order under refinement.
9. **Odd symmetry** `GZ(-phi) = -GZ(phi)`.
10. **Negative controls**: `KG > KM` gives negative initial stability; a factor-4
    `BM` typo is detected at large heel; degenerate inputs fail closed.

## Worked numbers (reference barge, `KG = 2.1 m`)

| heel [deg] | `GZ` exact [m] | `GZ` wall-sided [m] | residual [m] | `KN` [m] | `xB` [m] | `zB` [m] | bilge emerged | deck immersed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|   0.0 | -0.000000 | +0.000000 | 3.95e-16 | -0.000000 | -0.0000 | 1.0000 | no | no |
|   5.0 | +0.429064 | +0.429064 | 5.00e-16 | 0.612092 | +0.5249 | 1.0230 | no | no |
|  10.0 | +0.867073 | +0.867073 | 4.44e-16 | 1.231734 | +1.0580 | 1.0933 | no | no |
|  15.0 | +1.323960 | +1.323960 | 8.88e-16 | 1.867480 | +1.6077 | 1.2154 | no | no |
|  18.0 | +1.612055 | +1.612055 | 1.33e-15 | 2.260990 | +1.9495 | 1.3167 | no | no |
|  18.5 | +1.661341 | +1.661363 | 2.26e-05 | 2.327681 | +2.0076 | 1.3359 | yes | no |
|  20.0 | +1.799339 | +1.811825 | 1.25e-02 | 2.517581 | +2.1720 | 1.3933 | yes | no |
|  30.0 | +2.391387 | +2.950000 | 5.59e-01 | 3.441387 | +2.9607 | 1.7548 | yes | no |
|  43.0 | +2.675309 | +5.120957 | 2.45e+00 | 4.107505 | +3.5688 | 2.1957 | yes | yes |
|  60.0 | +2.341923 | +12.037753 | 9.70e+00 | 4.160576 | +3.8750 | 2.5670 | yes | yes |
|  90.0 | +0.900000 | +800128136434071378374934856204288.000000 | 8.00e+32 | 3.000000 | +4.0000 | 3.0000 | no | no |

## Note on the deck-immersion estimate

`wall_sided_validity_deg` returns two closed-form angles derived under the
wall-sided assumption `h = T`.  Only the smaller (binding) one is physical.
When `D > 2 T` the keel edge dries out first, the waterline height then drops
below `T`, and the real deck immersion happens *later* than the estimate: for the
reference barge the estimate is 33.69 deg while the geometric
solver first flags the event at 37.00 deg, the first sample of the stored 0.5 deg
table (gap > 3 deg; a 0.1 deg scan brackets it at 36.9 deg).  Use the
`deck_immersed` flags for the true event; the geometry solver is required
there, the closed form is not available.

## Deliberate limitations

Supported: prismatic rectangular sections, free heel at fixed displacement and
zero trim, 0-90 deg, arbitrary `KG`, cross-curves, dynamic-stability areas.

Not supported: arbitrary hull forms (no section tables / Bonjean integration
yet), trim and free-trim coupling, free-surface correction, flooding after deck
submersion (the solver keeps a closed box, so the post-deck-immersion branch is a
mathematical continuation, not a survivability prediction), damage stability,
seakeeping, and any classification-society rule check.  Passing the tests is not
design approval.

## Reproduce

```bash
PYTHONPATH=src python -m pytest tests/test_marine_large_angle_gz.py -q
PYTHONPATH=src python scripts/plot_large_angle_gz.py   # regenerates evidence JSON + figure
```
