# Marine-panel performance and truthfulness gate

`execute_panel_performance_gate` prevents an expensive full qualification
matrix from hiding a solver scalability failure.  It executes one accepted
arc-length point on a 2x2 diagnostic mesh and then a 4x4 qualification mesh.
The second stage is not started unless the first passes.

The default limits are 30 seconds for 2x2 and 120 seconds for 4x4.  Each stage
must finish one point before its wall limit, satisfy a relative free-DOF
equilibrium norm of at most `1e-6`, and record a SHA-256 digest of the full
accepted displacement, load factor, and every material-point plastic state.
Timeouts, singular systems, rejected paths, missing state evidence, and
imbalanced points fail closed.

Before timing begins, the runner also assembles the zero-displacement virgin
state and checks the internal-force norm on free DOFs against the gross-section
squash force.  This distinguishes a slow solver from an invalid continuation
origin.  The residual-stress surrogate fails the raw check because a zero
global longitudinal resultant does not imply a self-equilibrated discrete
nodal projection.  The dense solver therefore performs a transactional initial
equilibrium solve.  The gate accepts that route only when the relaxed residual
passes the same tolerance and both relaxed displacement and complete material
state are protected by a deterministic SHA-256 digest.

The current real dense run passes both stages.  The 2x2 first point took about
3.35 seconds and the 4x4 point about 13.06 seconds, versus limits of 30 and 60
seconds.  Their maximum relative free-DOF equilibrium norms were respectively
`1.33e-13` and `2.04e-13`.  A subsequent three-point pre-peak run completed in
about 6.00 and 23.47 seconds.  Loads remained increasing and all material
points remained elastic; these three points therefore establish continuity,
not a peak or post-peak result.

This is deliberately a performance and first-equilibrium-point gate.  Passing
it does **not** qualify ultimate strength.  Peak and descending response,
4/8/12 mesh convergence, continuation-step convergence, and energy balance
remain separate downstream requirements.  The 2x2 mesh is never used as
qualification evidence.

The machine-readable report is written to `performance-gate.json`, while each
underlying job retains the existing resumable evidence record.  Run it with:

```python
from tensorfem.marine_panel_execution import execute_panel_performance_gate

report = execute_panel_performance_gate("panel-evidence", resume=False)
```
