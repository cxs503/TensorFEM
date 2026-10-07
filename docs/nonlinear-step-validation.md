# Unified nonlinear load-step validation

`nonlinear_step.py` supplies transactional material state, adaptive load
increments, full or modified Newton iteration, residual backtracking, rejected
increment rollback, convergence history and portable JSON restart state.

## Qualification cases

| case | reference | acceptance |
|---|---|---:|
| Total-Lagrangian single bar | exact Green-strain equilibrium, `P=EA/2[(1+u/L)^3-(1+u/L)]` | displacement error `<3%` |
| symmetric two-bar shallow arch | exact truss geometry and first limit point | peak-load error `<3%` |
| bilinear elastoplastic bar | exact uniaxial return-mapping response | displacement error `<3%` |

The automated results are substantially tighter than 3%. Tests also force a
rejected large increment and verify rollback, restart serialization, and a
fail-closed exception after minimum-increment exhaustion.

## Scope boundary

This qualifies the load-step controller with Total-Lagrangian trusses and a
uniaxial elastoplastic bar. It does **not** qualify general 3-D finite-strain
plasticity, surface contact, arc-length continuation, or post-buckling.
