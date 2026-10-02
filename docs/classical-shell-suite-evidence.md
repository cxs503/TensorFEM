# Versioned classical-shell evidence suite

## Suite contract

`shell_benchmark_suite.py` packages four independently versioned cases. Each
case retains its own definition/evidence hashes; the sorted suite archive adds
suite ID, suite version, tier, aggregate status and a total SHA-256. Duplicate,
unsorted, missing, modified or failed cases invalidate the suite. Drift reports
compare every common mesh inside every case after verifying both total and
per-case hashes.

Quick evidence is the default CI tier. Full release evidence is enabled with
`TENSORFEM_FULL_SHELL_BENCHMARKS=1`; it reruns all 19 meshes and currently
takes about 19 seconds on the reference CPU. Thus ordinary CI verifies schema,
hashes, status and recorded reproduced values without repeatedly paying the
nonlinear pure-bending cost.

## Cases and sources

### Scordelis--Lo Roof

Full cylindrical roof: length 50, radius 25, half-angle 40 degrees, thickness
0.25, `E=4.32e8`, `nu=0`, downward surface load 90 and diaphragm ends. Target
is midspan free-edge vertical displacement `-0.3024`.

Source: MacNeal and Harder (1985), DOI
`10.1016/0168-874X(85)90003-4`.

| mesh | error | role/status |
|---:|---:|---|
| 2 | 3095.019% | convergence fail |
| 4 | 28.330% | convergence fail |
| 6 | 5.588% | convergence fail |
| 8 | 1.875% | qualification pass |
| 12 | 0.339% | qualification pass |
| 16 | 0.023% | qualification pass |

Capability boundary: linear metric-consistent cylindrical Q4, not a general
arbitrary-curvature nonlinear shell.

### Pinched Cylinder

Symmetry octant of a length-600, radius-300, thickness-3 cylinder with
`E=3e6`, `nu=0.3`, rigid diaphragm and quarter unit pinch load. Reference
radial displacement is `-1.8248e-5`, from the same MacNeal--Harder DOI.

| mesh | error | role/status |
|---:|---:|---|
| 3 | 1.420% | convergence pass, not qualification |
| 4 | 2.857% | convergence pass, not qualification |
| 6 | 5.374% | convergence fail |
| 8 | 3.350% | convergence fail |
| 12 | 0.985% | qualification pass |

The nonmonotone coarse regime remains visible; fortuitous coarse passes do not
replace the 12x12 qualification.

### Hemispherical Shell with 18-degree Hole

Radius 10, thickness 0.04, `E=6.825e7`, `nu=0.3`, quarter symmetry and
alternating unit equator loads. Reference radial displacement magnitude is
`0.0924`, again from MacNeal--Harder. Errors for 4/6/8/12/16 are respectively
`33.779%`, `18.767%`, `6.952%`, `0.690%`, `0.869%`; only 12 and 16 qualify.

Capability boundary: linear projected MITC-like Q4 quarter model, not general
nonlinear doubly curved response.

### Large-rotation Pure Bending

Length-10, width-1, thickness-0.1 strip with `E=1.2e6`, `nu=0`, clamped root
and prescribed 90-degree end rotation. Circular-elastica tip coordinate is
`20/pi`. Errors for 1/2/4 elements are `11.072%`, `2.617%`, `0.645%`; 2 and 4
qualify.

Related source: Bathe and Bolourchi (1979), DOI
`10.1002/nme.1620140703`. The numerical reference is the explicit circular
formula, not a digitized plot. Boundary: rotation-controlled geometric
nonlinearity with linear elasticity, not load-controlled post-buckling.

## Current archive

The quick suite total hash is
`11a6ed5b9e6cac9b81609011171274e82453c197344348a24289059d180a28c9`.
All formal fine meshes are strictly below 3%; all stated coarse failures remain
in the archive. Any formal failure raises instead of producing a passing suite.

```bash
PYTHONWARNINGS=error PYTHONPATH=src python -m pytest -q \
  tests/test_shell_benchmark_suite.py
TENSORFEM_FULL_SHELL_BENCHMARKS=1 PYTHONWARNINGS=error PYTHONPATH=src \
  python -m pytest -q tests/test_shell_benchmark_suite.py
```
