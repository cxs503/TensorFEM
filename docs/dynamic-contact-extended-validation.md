# Edge CCD, unified dynamic contact and frictional impact

## Discrete scope

`dynamic_contact_extended.py` adds conservative edge--edge CCD for two line
segments whose four endpoints follow linear trajectories during a time step.
It shares a deterministic event queue with vertex--moving-TRI3 CCD: both event
types are sorted by earliest TOI and deterministically deduplicated.

Swept edge AABBs are inserted into a uniform spatial hash. Cell contents and
the final candidate set are sorted, giving reproducible broad-phase output.
Incident edges are excluded. This is a scalable prototype rather than a
production BVH with incremental refitting.

## Impact law

At an event, mass-lumped effective mass produces a zero-or-user-restitution
normal impulse. The tangential sticking impulse is projected onto the Coulomb
disk `||Jt|| <= mu Jn`. Equal and opposite weighted nodal impulses preserve
linear momentum; coincident closest points preserve angular momentum. The
implementation audits kinetic energy and fails if an admissible impact update
increases it.

## Verification

- Perpendicular edges separated in Z move through each other at high speed.
  Computed TOI is `0.49999999995` against exact `0.5`, approximately `1e-10`
  relative error.
- Splitting the step at global time `0.4` reproduces the same global TOI to
  below `1e-8`.
- A Coulomb impact with `mu=0.3` gives `Jn=1`, `||Jt||=0.3`; total impulse and
  angular impulse are zero to round-off. Kinetic energy decreases from `2.0`
  to `1.245`.
- A spatial-hash case containing 40 remote edges and one crossing pair returns
  exactly one sorted pair on repeated calls.
- Transverse coplanar edges have the unique cross-product normal and are
  accepted at `TOI=0`. Parallel/collinear overlap has no unique normal and
  raises `AmbiguousCoplanarContactError` rather than choosing an arbitrary
  direction.

All TOI, momentum, angular-momentum and energy contracts are far within 3%.

## Remaining boundaries

Only linear endpoint trajectories and straight edges/TRI3 faces are covered.
Parallel coplanar overlap resolution, curved trajectories, curved geometry,
edge-face persistent manifolds, simultaneous-impact complementarity and
frictional heating are not implemented. The sequential impulse kernel is not
a general multi-contact impact solver.

```bash
PYTHONWARNINGS=error PYTHONPATH=src python -m pytest -q \
  tests/test_dynamic_contact_extended.py
```
