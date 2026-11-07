# fem_beam_static — Euler–Bernoulli beam statics (P19-N01 rung L1)

Verdict: **PASS** (see `gates_verified.json`)
Driver: `run.py` · Evidence: `result.json`, `gates_verified.json` · Figures: `deflection_profile.png`, `convergence.png`

Model: 2-node cubic Hermite beam (4 dof per element, `w` and `w_,x`), consistent
nodal load `q*[le/2, le²/12, le/2, -le²/12]`, `N = 10/20/40` elements.
Full derivation and discussion: `docs/fem-verification-ladder-p19.md`.

| case | exact solution | finest-mesh error |
|---|---|---:|
| SS beam, `q = 1 N/m`, `L = 10 m`, `EI = 1000 N m²` | `w(x)=q x(L³-2Lx²+x³)/(24EI)` at 5 off-node sections | 2.5e-07 (worst) |
| cantilever, tip load `P = 0.1 N` | `w=P L³/(3EI)`, `θ=P L²/(2EI)` | 6.5e-11 / 6.2e-11 |

| N | h [m] | max section error | L2 field error |
|---:|---:|---:|---:|
| 10 | 1.00 | 2.67e-05 | 1.78e-05 |
| 20 | 0.50 | 1.81e-06 | 1.11e-06 |
| 40 | 0.25 | 2.54e-07 | 6.97e-08 |

Observed orders: L2 field **4.00**, section errors **3.36** (threshold ≥ 2).

Why the sections are *off-node*: nodal displacements of this element are exact for
Euler–Bernoulli distributed loads, so grading only nodal values would prove
nothing about the shape functions; the five sections (x = 0.9, 2.4, 3.9, 5.4,
6.9 m) are interior points for all three meshes.

Limitations: Euler–Bernoulli kinematics only (no shear, no rotary inertia, no
warping); 1-D idealisation of the beam; the cantilever case is round-off limited
so it verifies the load vector and boundary conditions, not the convergence rate.

```
PYTHONPATH=src python benchmarks/verified/fem_beam_static/run.py
pytest tests/test_p19_ladder_artifacts.py -q
```