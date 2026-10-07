# Colored frictional contact graph

## Batched Coulomb update

`colored_frictional_contact.py` extends the deterministic conflict coloring
with a two-dimensional tangent block at every contact. A stable orthonormal
tangent basis is constructed from the contact normal. The batched 2x2
Delassus block includes mass, contact-point lever arms and world inertia for
both bodies. A sticking trial is solved in that block and projected onto
`||Jt|| <= mu*Jn` when sliding.

Normal impulse, world-vector tangential impulse and stick/slip flag form an
immutable warm-start state. Returning a trial does not mutate the checkpoint,
so rejection and rollback remain explicit. All tensors use the device of the
prepared colored data; CPU and CUDA follow the same operations.

## Verification

- An eight-body high-coupling chain includes normal, tangential and angular
  coupling. Colored velocities and angular velocities match scalar rigid PGS
  within `2e-7`; complementarity is below `1e-7`, cone residual below `1e-10`
  and total kinetic energy is nonincreasing.
- Separate high- and low-friction cases exercise sticking and sliding history.
  Warm and cold starts converge to the same state, while discarding the trial
  leaves the input checkpoint unchanged.
- 1024 independent eccentric ground contacts occupy one color. Colored and
  scalar paths agree in both linear and angular velocity to `1e-11`, with zero
  complementarity and cone residual. A representative CPU run took `0.0088 s`
  for the colored path and `1.218 s` for scalar PGS (about 138x). These values
  characterize this independent-contact case only and are not a general speed
  guarantee.
- The 1024-contact case converges in two sweeps and reports total inelastic plus
  frictional kinetic-energy loss `552.96`; energy never increases.

Every comparison and residual is below the 3% gate.

## Limits

Colors remain sequential and only constraints within one color are batched.
Strongly coupled frictional graphs can need substantially more sweeps than
independent constraints. The current energy output is total collision loss,
not a thermodynamic split between restitution and frictional heat. Actual GPU
performance, multi-GPU scheduling, nonlinear friction and exact cone LCP are
not claimed.

```bash
PYTHONWARNINGS=error PYTHONPATH=src python -m pytest -q \
  tests/test_colored_frictional_contact.py
```
