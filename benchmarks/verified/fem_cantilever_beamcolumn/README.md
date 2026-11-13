# fem_cantilever_beamcolumn -- CASE-CANTILEVER-TIP axial-preload rung

Verdict: **PASS** (`gates_verified.json`, 26/26 gates).
Driver: `run.py` · Docs: `docs/CASE-CANTILEVER-TIP-beamcolumn-amplification.md` · Figures:
`docs/assets/cantilever_beamcolumn/` · Evidence: `result.json`, `gates_verified.json`.

Why this anchor exists: every other rung of CASE-CANTILEVER-TIP holds the internal axial force at
zero, and `tensorfem.buckling`'s geometric stiffness was only ever consumed by an eigenvalue solve,
which fixes just the ratio of `Kg` to `K` along the buckling mode.  Here the *static* load path
`(K + N Kg) u = f` is graded at finite preload against the exact amplification

    AF_t(u) = 3 [u - tanh u] / u^3      (tension,  < 1)
    AF_c(u) = 3 [tan u  - u ] / u^3     (compression, > 1, pole at u = pi/2)

with `u = L sqrt(|N|/EI)`, the oracle derived in sympy from the ODE (residual
`0`) and cross-checked by the analytic continuation
`AF_c(i u) = AF_t(u)` (residual 0.0e+00) and by the pole sitting on
the fixed-free Euler load (0.0e+00 relative).

| statement | result |
|---|---|
| FE vs exact, physically admissible rungs (`n >= 4`) | worst 0.0309 % (`fe_vs_exact_compression_u1.5`) -- inside the 3 % case-pool tolerance |
| convergence order of the nodal tip deflection | ~3.99 (tension) / 4.00 (compression near `N_cr`) |
| critical load from the same `K`, `Kg` (n = 32) | 0.0000 % vs `pi^2 E I/(4 L^2)` |
| one element at `u = 1.5` compression | 6.46 % error -- graded as a *required* failure |
| preload effect at `u = 1` | deflection x 1.672 (compression), x 0.715 (tension) |

Reproduce:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src python benchmarks/verified/fem_cantilever_beamcolumn/run.py
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src python benchmarks/verified/fem_cantilever_beamcolumn/render_docs.py
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src pytest tests/test_cantilever_beamcolumn_anchor.py -q
```
