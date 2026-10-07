# TensorFEM-only marine structural qualification

This gate deliberately excludes TensorLBM, CFD, resistance, propulsion,
free-surface flow and fluid-structure coupling. It answers a narrower question:
which structural mechanics needed by ship and offshore applications have direct,
executable TensorFEM evidence?

The report combines 15 non-zero scalar references with the four-case classical
shell suite (Scordelis--Lo roof, pinched cylinder, hemispherical shell and
large-rotation pure bending). The classical cases preserve coarse-mesh failures
instead of hiding them, and accept only designated qualification meshes below
the 3% threshold.

| Area | Current qualification |
|---|---|
| global hull/beam static response | qualified for the stated linear benchmarks |
| plates and shells | qualified for patch, bending and four classical cases |
| structural modal response | qualified for the cantilever reference |
| buckling | beam/column qualified; local plate buckling qualified prototype |
| plasticity | qualified prototype at material and limited global TET4 levels |
| local stiffened-plate buckling | qualified prototype for Navier/finite-difference ideal plates |
| hull-girder ultimate strength | qualified section prototype; not full progressive collapse |
| fatigue assessment primitives | hot-spot/S-N/Miner hand-oracle qualified prototype |
| imperfect plate postbuckling | single-mode reduced-order qualification only |
| multi-component hull yielding | section prototype; not shell progressive collapse |
| spectrum fatigue and crack growth | rainflow/Miner/Paris primitives qualified |
| full shell progressive collapse | not qualified |
| Shell4 initial-stress eigenbuckling | qualified against Navier plate load |
| imperfection/residual-stress plasticity | reduced-order strip prototype |
| ResultDB hotspot and LEFM | qualified post-processing primitives |
| shell arc-length postbuckling | not qualified; algorithm gate alone is insufficient |

The next TensorFEM-only evidence should therefore target local plate buckling,
stiffened-panel collapse, nonlinear hull-girder ultimate strength, wet-free
structural vibration baselines, welded-detail hotspot stress and fatigue damage.
