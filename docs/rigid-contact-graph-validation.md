# Rigid-body inertia and sparse contact islands

## Formulation

`rigid_contact_graph.py` represents each body by centre of mass, mass, world
inertia tensor, linear velocity and angular velocity. Contact-point velocity is
`v + omega x r`. The 3x3 Delassus operator includes translational inverse mass
and `-skew(r) I^-1 skew(r)` rotational compliance for both bodies.

Projected Gauss--Seidel solves normal complementarity and the Coulomb disk.
Every impulse updates both linear and angular velocity. Stable contact keys
provide deterministic order and warm starts; returned history remains a trial
state until explicitly committed.

## Sparse islands

A deterministic union--find decomposes the body/contact graph. Fixed ground
is deliberately not a graph vertex, so unrelated ground contacts remain
independent islands suitable for parallel execution. Each island is solved in
stable key order. The current Python implementation executes islands
sequentially but exposes their deterministic partition for future workers.

## Verification

- An eccentric frictional impact between two finite-inertia bodies produces
  nonzero angular velocities `-0.7742` and `-0.5161`. Total linear and angular
  momentum errors are below `1e-11`, the friction-cone residual is zero and
  complementarity residual is `2.87e-11`. Kinetic energy decreases from `5`
  to `3.5484`.
- Two disconnected two-body islands solved together reproduce their separate
  solutions to `1e-12`.
- A warm start converges to the cold solution without more sweeps, and the
  virgin state remains unchanged when the trial state is discarded.
- The performance regression contains 1000 bodies, 1000 constraints and 1000
  independent ground-contact islands. All final closing velocities and
  complementarity residuals are below `1e-10`; the automated ceiling is 10 s
  on the reference CPU rather than a claimed universal throughput figure.
- Singular inertia, invalid body indices/normals and duplicate keys fail
  closed.

All conservation, cone, complementarity and energy errors are below 3%.

## Limits

The prototype uses world-frame inertia supplied for the current configuration;
orientation integration and inertia rotation belong to the time integrator.
Large coupled islands still use serial PGS and may converge slowly. Graph
coloring, block solvers, GPU islands, rigid-joint coupling and exact cone LCP
are not implemented. Ambiguous collinear CCD normals remain fail-closed.

```bash
PYTHONWARNINGS=error PYTHONPATH=src python -m pytest -q \
  tests/test_rigid_contact_graph.py
```
