# j2_elastoplastic_three_case

Three-case qualification of the J2 (von Mises) elastoplastic surface with
**linear isotropic hardening**: one material card, one code path, three
independent oracles.

Campaign node `p337w1-n07-fem-elastoplastic`. Report:
[`docs/j2-elastoplastic-three-case.md`](../../docs/j2-elastoplastic-three-case.md).
Module: `tensorfem.j2_elastoplastic_three_case`.

## What is verified

| case | problem | oracle | criterion (band) |
|---|---|---|---|
| A | TET4 tensile bar `L=100 mm`, `A=10×10 mm`, `ε = 0…5 %` | bilinear material card + independent scalar bisection | `σ(ε)` < 5 %, `σ_y` < 2 % |
| B | material-point pure shear (thin-wall tube `R=50 mm`, `t=2 mm`) + TET4 patch test | closed form `τ(γ)`, `τ_y = σ_y/√3` | `τ_y` < 2 % |
| C | plane-strain axisymmetric tube `R_i=50`, `R_o=100 mm` under internal pressure | Hill 1950 closed form; `u_r(r)` by quadrature | `r_p(p)` < 5 % (5 levels), `u_r(r)` < 10 % |
| C6 | yield surface in the π plane, 6 directions | `q = σ_y` on a circle | < 1 % |
| C7 | `h`-refinement `n_r = 16, 32, 64` | Roache GCI on `u_r(a)` | GCI reported |

## Run

```bash
python benchmarks/j2_elastoplastic_three_case/run.py           # full, ~4-5 min
python benchmarks/j2_elastoplastic_three_case/run.py --quick   # smoke, ~1 min
python benchmarks/j2_elastoplastic_three_case/render_markdown.py  # refresh doc tables
```

Outputs (committed as evidence):

* `artifacts/j2_elastoplastic_three_case/report.json` — every number, plus the
  criteria table with `passed` flags;
* `fig1..fig5` PNGs in the same directory, regenerated from that JSON.

## Notes a reader must not skip

* The campaign's literal cylinder band `p = 0…100 MPa` is **elastic** for this
  card (`p_y = 108.25 MPa`); it is reported as an elastic check and the plastic
  study is run at `p/k = 1.00 … 1.35` (`144 … 195 MPa`, below collapse
  `p_lim = 200.09 MPa`).
* No ASTM E8 experimental dataset is in the repository, so the uniaxial chord is
  scored against the analytic card; the experimental comparison is *withheld*,
  not passed.
* The residual ≈1.4 % (`r_p`) / ≈2 % (`u_r`) offset against Hill at the highest
  pressure is the classical plane-strain `σ_z` inconsistency for `ν ≠ 1/2`, not
  mesh error: the GCI of the same quantity is ≤ 0.1 %.