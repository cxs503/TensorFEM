# fem_curved_ring_solid -- plane-stress continuum oracle for the quarter-ring benchmark

**Anchor**: fem_curved_ring_solid · **Mission**: m14873 · **Verdict**: PASS
(8/8 gates)

The quarter-ring curved cantilever in `tensorfem.curved_beam_benchmark` is graded
by the shell elements against a curved-beam *theory* closed form.  This anchor
supplies an oracle from a different theory family: the same ring as a 2-D
plane-stress continuum with Q8 elements (`tensorfem.curved_ring_continuum`),
refined and Richardson-extrapolated, plus a surface hoop-stress grading against
the Winkler--Bach curved-beam profile.

| | |
|---|---|
| module | `src/tensorfem/curved_ring_continuum.py` |
| driver | `run_verification.py` |
| tests | `tests/test_curved_ring_continuum.py` |
| doc | `docs/curved-ring-continuum-oracle.md` |
| figures | `docs/assets/curved_ring_continuum/` |
| reproduce | `python benchmarks/verified/fem_curved_ring_solid/run_verification.py` |

Headline numbers (see `result.json` for the full evidence):

- tip compliance vs closed form at t/R = 0.05: **0.108 %** tangential,
  **0.054 %** radial (Richardson-extrapolated);
- closed-form drift `0.14 (t/R)^2.50` %, i.e.
  0.72 % at t/R = 0.30;
- surface hoop stress vs Winkler--Bach: <= 0.85 %;
- the flat-beam `N/A + M y/I` check under-reports the inner fibre by
  10.47 % at t/R = 0.30.

Limitations are listed in `gates_verified.json` (`limitations`).
