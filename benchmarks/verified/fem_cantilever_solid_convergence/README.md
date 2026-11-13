# fem_cantilever_solid_convergence -- CASE-CANTILEVER-TIP

Verdict: **PASS** (`gates_verified.json`).
Driver: `run.py` · Evidence: `result.json`, `gates_verified.json` · Figures:
`docs/assets/cantilever_solid_conv/` · Full discussion:
`docs/CASE-CANTILEVER-TIP-solid-convergence.md`.

Closes the verification debt of `benchmarks/verified/fem_cantilever_shear/`: its solid residual
(0.4 % too stiff, "only 0.63 of the shear compliance", loose `>= 0.5` gate) was measured against
beam oracles only.  Here the HEX20 continuum is driven with the exact parabolic tip traction and
graded against the library's exact plane-stress elasticity field, while the root boundary
condition is varied as a controlled experiment.

| question | answer |
|---|---|
| does the solid agree with an elasticity (not beam) oracle? | yes, worst rung `1.792 %` |
| is the observable mesh-converged? | yes, cross 4x4 -> 6x6 `0.001 %`, axial 20 -> 40 `0.002 %` |
| who owns the 0.4 % stiff bias? | the fully clamped root; freeing lateral contraction moves the tip `+0.566 %` at `L/h = 100` |
| why could the old shear gate not be tightened to `kappa = 5/6`? | the exact 2-D field implies `kappa_equivalent = 0.6667`, not 0.8333 |

```bash
OMP_NUM_THREADS=8 PYTHONPATH=src python benchmarks/verified/fem_cantilever_solid_convergence/run.py
PYTHONPATH=src pytest tests/test_cantilever_solid_convergence.py -q
```
