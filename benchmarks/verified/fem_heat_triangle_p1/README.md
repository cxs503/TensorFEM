# fem_heat_triangle_p1 — CASE-FEM-HEAT-L2 (round 2, P1 triangles)

Steady heat conduction verified with manufactured solutions on **P1 linear
triangles**: L2 rate 2 / H1 rate 1, on two diagonal splits and on a distorted
triangulation, with a mixed Dirichlet/Neumann case for the flux path and a
scikit-fem `ElementTriP1` cross-validation.

This study complements — does not repeat — `benchmarks/verified/fem_heat_conduction_l2`
(Q4/bilinear, structured rectangles), which is limited to one element family and
one topology; the manufactured fields are imported from
`tensorfem.heat_l2_convergence`.

| file | content |
|---|---|
| `run.py` | one-command reproduction: ladders, gates, figures, `result.json`, `gates_verified.json` |
| `result.json` | every ladder, norm, gate value and the parameter study |
| `gates_verified.json` | machine-checkable gate manifest (`tensorfem.verified-gates.v1`) |
| `make_report.py` | renders `docs/CASE-FEM-HEAT-L2-p1-triangle-round2.md` and publishes the platform report |

```
PYTHONPATH=src OMP_NUM_THREADS=4 python benchmarks/verified/fem_heat_triangle_p1/run.py      # ~47 s, 11/11 gates PASS
PYTHONPATH=src OMP_NUM_THREADS=4 python -m pytest tests/test_heat_l2_triangle.py -q           # 13 tests
PYTHONPATH=src python benchmarks/verified/fem_heat_triangle_p1/make_report.py                # report + figures
```

Headline (finest mesh, `nx = ny = 64`, dense 4225 unknowns):

* L2 rate `1.9802` (theory 2), H1 rate `1.000` (theory 1);
* `max|e|/max|T| = 8.33e-03` (< 3%);
* scikit-fem cross-validation: nodal difference `1.7e-15`, matrix difference `0.0`;
* post-processing-rule change under 3-level subdivision `3.2e-10`.

Limits: rectangle, uniform isotropic refinement, regular solutions, `k = 1`,
dense assembly (`nx <= 64`). The mixed-BC case is still pre-asymptotic
(1.91 at the finest pair) and is reported as supplementary evidence only.
