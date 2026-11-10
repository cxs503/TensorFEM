# Strong-coupling FSI: LBM fluid x modal FE beam

Status: **honest partial** (see `docs/reports/P153W2-N03_fsi_lbm.md`).

## Modules

| module | role | evidence |
|---|---|---|
| `src/tensorfem/lbm_channel2d.py` | D2Q9 BGK channel solver: half-way bounce-back walls/obstacles, Guo forcing, momentum-exchange force, Peskin IBM kernel for generic marker sets | REAL (validated on Turek-Hron 2D-2: CD and St within 11%, lift amplitude within 19%) |
| `src/tensorfem/fsi_lbm_strong_coupling.py` | partitioned Dirichlet-Neumann strong coupling, Aitken Delta^2 relaxation, moving bounce-back body, FE-consistent load projection, energy ledger | REAL (interface resultant conserved to 0.23%; sub-iteration residuals recorded) |

## Why the old "verified" FSI modules are MOCK

* `fsi_turek_hron.TurekHronFSI2Solver` forces the beam with a harmonic load
  whose frequency (`f_vortex = 3.0 Hz`) and amplitude (`F0 = 17.5 N`) are
  annotated in the source as *tuned to match the reference*. Its configured
  inflow (0.2 m/s) with `nu = 1e-3` gives Re = 30, while the docstring and the
  reported benchmark claim Re = 100.
* `theodorsen.find_critical_flutter` returns the closed form
  `U_cr = 2 sqrt(mu)`; comparing that value with `2 sqrt(mu)` is a tautology.
  `theodorsen.fsi_time_march` synthesises `exp(growth t) cos(omega t)` from the
  same formula instead of integrating the equations of motion.

## How to reproduce

```bash
PYTHONPATH=src python benchmarks/fsi_strong_coupling/run_validation.py \
    --case structure          # analytic beam checks (seconds)
PYTHONPATH=src python benchmarks/fsi_strong_coupling/run_validation.py \
    --case interface          # load-projection conservation (seconds)
PYTHONPATH=src python benchmarks/fsi_strong_coupling/run_validation.py \
    --case cylinder --n-steps 8000     # 2D-2 anchor (~80 s at D=20)
PYTHONPATH=src python benchmarks/fsi_strong_coupling/run_validation.py \
    --case fsi2 --cells 20 --u-lat 0.05 --duration 0.3 --zeta 0.02
PYTHONPATH=src python benchmarks/fsi_strong_coupling/make_report.py
PYTHONPATH=src python benchmarks/fsi_strong_coupling/make_acceptance.py
```
