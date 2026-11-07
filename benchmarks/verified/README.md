# `benchmarks/verified/` — campaign verification anchors

Each subdirectory is one **self-contained verification anchor**: an independent
closed-form oracle, a refinement ladder, graded gates and its own figures. The
anchors were created for campaign node **IN36-N01** (TensorFEM linear elasticity /
contact / modal verification surface).

| directory | anchor | oracle |
|---|---|---|
| `fem_cantilever/` | slender cantilever, `L/b/h = 10/1/0.1 m`, tip load | `w = F L^3/(3 E I)`, `sigma_xx = F(L-x)(z-h/2)/I` |
| `fem_hertz_contact/` | two steel spheres `R = 50 mm` under `F = 10 kN` | `a = (3FR/4E*)^(1/3)`, `p_0 = 3F/(2 pi a^2)` |
| `fem_plate_modal/` | SS rectangular plate `1.2 x 0.8 x 0.012 m` | `f_mn = (pi/2) sqrt(D/(rho h))[(m/a)^2+(n/b)^2]` |

Height of evidence per anchor:

```
result.json           every number, per refinement level (generated, never hand-edited)
gates_verified.json   schema tensorfem.verified-gates.v1: per gate the reference formula,
                      the value, the threshold and the verdict; plus limitations
*.png                 field / mode-shape / pressure figures and convergence plots
run.py                the driver (its module docstring states the physics and the model)
_cache/               optional per-level cache so a slow mesh can be solved in its own call
```

Reproduce:

```bash
OMP_NUM_THREADS=16 PYTHONPATH=src python benchmarks/verified/fem_cantilever/run.py
OMP_NUM_THREADS=32 PYTHONPATH=src python benchmarks/verified/fem_hertz_contact/run.py
OMP_NUM_THREADS=16 PYTHONPATH=src python benchmarks/verified/fem_plate_modal/run.py
PYTHONPATH=src python benchmarks/verified/render_report.py           # refresh the doc tables
PYTHONPATH=src pytest tests/test_verified_fem_anchors.py             # oracles + gate consistency
```

Notes that apply to all three:

* `_harness.py` holds only book-keeping (JSON, Richardson/GCI arithmetic, gate
  schema, matplotlib setup); no physics lives there, so a single anchor can be
  audited without reading the others.
* Gate values are cross-checked against `result.json` by the test suite, so an
  edited evidence file fails the tests instead of passing silently.
* Where a quantity does not converge monotonically, the evidence says so; see
  `fem_hertz_contact/result.json:convergence` and
  `docs/verified/fem-verification-anchors.md` §6.
## P19-N01 verification ladder (2026-11)

Three new anchors extend the structural verification face beyond the single-rung
cases above; each is self-contained (`run.py` + `result*.json` +
`gates_verified.json` + figures) and graded against a closed-form / series
oracle with a three-level mesh ladder.

| anchor | physics | oracle | data points | worst error |
|---|---|---|---:|---:|
| `fem_beam_static/` | Euler–Bernoulli beam: SS uniform load, cantilever tip load | `w(x)=q x(L³-2Lx²+x³)/(24EI)`, `PL³/(3EI)`, `PL²/(2EI)` | 7 | 2.5e-07 |
| `fem_plate_static/` | Kirchhoff plate: SSSS + CCCC under uniform pressure | Navier series (11 and 301 harmonics), Timoshenko tables 8/35 | 6 | 7.7e-03 |
| `fem_modal/` | SS beam modes 1–5 and SS plate modes (1,1)…(2,2) | `(nπ/L)²√(EI/ρA)`, Navier plate eigenfrequencies | 10 | 1.6e-05 |

`fem_modal/result.json` remains the legacy cantilever rung produced by
`benchmarks/structural_ladder.py`; the P19 modal ladder writes
`result_p19.json` alongside it. Report: `docs/fem-verification-ladder-p19.md`,
artifact tests: `tests/test_p19_ladder_artifacts.py`, element tests:
`tests/test_plate_bfs.py`.
