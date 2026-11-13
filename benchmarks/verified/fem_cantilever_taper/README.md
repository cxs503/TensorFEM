# fem_cantilever_taper

Anchor `CASE-CANTILEVER-TIP`, taper axis: a linearly tapered cantilever (`h(x) = h0 (1 - c x/L)`) graded against the closed-form Castigliano flexibility `B(c) = [-c - c^2/2 - ln(1-c)]/c^3` and the logarithmic shear factor `S(c) = -ln(1-c)/c`.

* driver: `run.py` (regenerates `result.json`, `gates_verified.json`, the doc and the figures)
* verdict: **PASS** (failed gates: none)
* generated: 2026-11-13T13:11:43+00:00

Run: `OMP_NUM_THREADS=4 PYTHONPATH=src python benchmarks/verified/fem_cantilever_taper/run.py`
