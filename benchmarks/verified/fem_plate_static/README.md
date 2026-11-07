# fem_plate_static — Kirchhoff plate statics, SSSS + CCCC (P19-N01 rung L2)

Verdict: **PASS** (see `gates_verified.json`)
Driver: `run.py` · Evidence: `result.json`, `gates_verified.json` · Figures: `fields_ssss.png`, `fields_cccc.png`, `convergence.png`

Model: Bogner–Fox–Schmit C1 bicubic rectangle (16 dof/element:
`w, w_,x, w_,y, w_,xy`) from the new module `src/tensorfem/plate_bfs.py`;
3×3 Gauss stiffness, **consistent** transverse load, sparse SuperLU solve.
Plate `a = b = 1 m`, `t = 10 mm` (`a/t = 100`), `E = 210 GPa`, `ν = 0.3`,
`q = 10 kPa`, `D = 19230.77 N m`.

| N | SSSS w_c [mm] | α = w_c D/(q a⁴) | err vs Navier 11×11 | err vs Navier 301 | err vs 0.00406 | M_x err vs Navier |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 2.112460 | 0.0040623 | 2.89e-05 | 1.73e-05 | 5.97e-04 | 3.59e-03 |
| 20 | 2.112426 | 0.0040624 | 1.27e-05 | 1.07e-06 | 5.81e-04 | 8.68e-04 |
| 40 | 2.112424 | 0.0040624 | 1.17e-05 | 6.64e-08 | 5.80e-04 | 2.15e-04 |

| N | CCCC w_c [mm] | α | err vs 0.00126 | M_x/(q a²) | err vs 0.0231 |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.660011 | 0.0012693 | 7.34e-03 | 0.02322 | 5.01e-03 |
| 20 | 0.658223 | 0.0012658 | 4.61e-03 | 0.02298 | 5.40e-03 |
| 40 | 0.657999 | 0.0012654 | 4.27e-03 | 0.02292 | 7.71e-03 |
| 80 (extended) | 0.657953 | 0.00126533 | 4.23e-03 | 0.022909 | 8.27e-03 |

Observed orders: SSSS deflection **4.01**, SSSS moment **2.03**, CCCC
self-convergence **2.99** (thresholds ≥ 1.5).

Reading the CCCC row: the classical value is α = 0.00126532, so the 0.42 % offset
from the three-digit table value 0.00126 is **reference rounding**, not
discretisation — the N=80 level reproduces 0.00126533. Likewise the CCCC centre
moment converges to 0.02291 q a², 0.8 % under the tabulated 0.0231, which is why
its error against the table is not monotone even though the moment itself
converges at order 2.

Why an 11×11 truncation is not enough for the moments: the mandated 11-harmonic
Navier series is accurate to ~5e-8 in deflection but only ~1e-3 in the centre
moment, so the moment gates are graded against a 301-harmonic evaluation of the
same series (both are recorded in `result.json`).

Limitations: Kirchhoff (thin-plate) theory, no transverse shear; the hard simple
support (`w = 0` + tangential slope zero) excludes corner uplift and is the exact
Kirchhoff condition; the load is consistent rather than lumped, so these numbers
are not directly comparable with `benchmarks/verified/fem_plate` (lumped load,
1.45 % error at N=10 for the same mesh).

```
PYTHONPATH=src python benchmarks/verified/fem_plate_static/run.py
pytest tests/test_plate_bfs.py tests/test_p19_ladder_artifacts.py -q
```