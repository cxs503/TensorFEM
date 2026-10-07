# Multi-contact complementarity and persistent manifolds

## Solver

`multi_contact_impulse.py` accepts all constraints at the earliest unified CCD
time and solves them together with deterministic projected Gauss--Seidel.
Normal accumulated impulses are projected onto `jn >= 0`; tangential impulses
are projected onto the Coulomb disk `||jt|| <= mu*jn`. The result reports
nonpenetration/complementarity residual, cone residual and kinetic energy.

Constraint keys provide stable deterministic ordering. Accepted normal and
friction impulses form an immutable warm-start state. A rejected global step
retains the input state unchanged. `constraints_from_unified_events` converts
the common vertex--face/edge--edge earliest-event set without introducing a
second event ordering.

## Persistent manifold

`build_vertex_face_manifold` retains every requested vertex or edge endpoint
within tolerance of a TRI3 face. It therefore represents a two-point edge--face
or multi-vertex face manifold rather than collapsing the contact to one point.
Consistent master barycentric weights preserve resultant impulse.

## Verification

- Two independent simultaneous points impacting at speed `2` receive exact
  normal impulses `[2,2]`; final normal velocities and complementarity
  residual are zero.
- Four points with velocity `(1,0,-1)` and `mu=0.25` each receive analytical
  impulses `(-0.25,0,1)`. Both complementarity and cone residuals are below
  `1e-12`; kinetic energy strictly decreases.
- A two-endpoint edge manifold against a TRI3 plane preserves total impulse
  with residual below `1e-12` and satisfies complementarity below `1e-9`.
- Warm and cold starts converge to identical velocities; warm start needs no
  more sweeps. Discarding the returned state leaves the virgin state empty.
- With constant pre-impact velocity, full and time-refined impact steps return
  identical impulses to `1e-13`.

All analytical impulse and residual errors are far below the 3% gate.

## Limits

This is a nodal translational, mass-lumped, small contact-set PGS solver.
Large dense contact graphs may converge slowly and no graph coloring or block
preconditioner is provided. Parallel/collinear contact normals remain
fail-closed in CCD. Rigid-body rotational inertia, shock propagation,
nonlinear friction coefficients and exact cone complementarity are outside the
current scope.

```bash
PYTHONWARNINGS=error PYTHONPATH=src python -m pytest -q \
  tests/test_multi_contact_impulse.py
```
