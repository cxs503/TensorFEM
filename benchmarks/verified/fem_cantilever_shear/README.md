# fem_cantilever_shear -- CASE-CANTILEVER-TIP shear-validity ladder

Verdict: **PASS** (`gates_verified.json`).
Driver: `run.py` · Evidence: `result.json`, `gates_verified.json` · Figures:
`docs/assets/cantilever_shear/` · Full discussion: `docs/CASE-CANTILEVER-TIP-shear-validity.md`.

Why this anchor exists: the original CASE-CANTILEVER-TIP rung (`L/h = 100`) has an
Euler-Bernoulli vs Timoshenko oracle gap of 7.80e-03 %, so it
cannot detect a missing shear degree of freedom. This ladder adds rungs down to
`L/h = 4`, where the oracles separate by
4.87 % (> 3 %), and grades three models of the library.

| model | oracle | verdict on the ladder |
|---|---|---|
| `frame_eb` (cubic Hermite, shear-rigid) | Euler-Bernoulli | exact to round-off; **fails** the Timoshenko oracle at `L/h = 4` (-4.65 %) |
| `beam_timoshenko` (kappa = 5/6) | Timoshenko | exact to round-off on every rung |
| `solid_hex20` (3-D continuum) | Timoshenko | within 1.71 % at the thick rung, recovers 0.63 of the shear compliance |

Follow-up (cross-section convergence): `cross_section_convergence.py` ·
`cross_section_convergence.json` · `docs/CASE-CANTILEVER-TIP-cross-section-convergence.md`.
It adds a certified sparse HEX20 solve (identical to the dense path to 3e-12, models up to
6.6e4 dof) and refines the cross-section on an isotropic family, which the dense path could
not afford. Outcome: the `< 3 %` verdicts of this bundle stand -- the `L/h = 4` residual
converges to -1.67 % vs Timoshenko, i.e. it is model difference, not mesh -- while the
shear-recovery number above (0.63) is shown **not** to be converged or rung-independent, so
it is not a measurement of `kappa`; and the slender-rung residual attributed below to the
clamped root is dominated by discretisation (it halves at `L/h = 20`,
-0.380% -> -0.180%, and changes sign at `L/h = 10`,
-0.146% -> +0.029%).

Reproduce:

```bash
OMP_NUM_THREADS=8 PYTHONPATH=src python benchmarks/verified/fem_cantilever_shear/run.py
PYTHONPATH=src pytest tests/test_cantilever_shear_anchor.py -q
```
