# Continuous collision detection and dynamic pair history

## Qualified discrete scope

`self_contact_ccd.py` implements conservative continuous collision detection
for a vertex and a linearly moving TRI3 face during one time step. Swept AABB
broad phase removes impossible pairs and incident vertex/face pairs.
Conservative advancement uses a relative-speed upper bound, so a fast vertex
cannot cross a thin triangle between the two discrete configurations.

The earliest time of impact (TOI) clips the accepted substep. A zero-restitution
normal impulse is distributed to the vertex and triangle nodes with
barycentric weights. This conserves total impulse and moment. The caller then
solves the remaining substep, rather than accepting the tunneled end position.

This scope does not include edge--edge CCD, curved trajectories, curved faces,
coplanar overlap resolution, frictional impact impulses or continuous broad
phase acceleration structures.

## Transactional history

Every active `(vertex, face)` pair records age, accumulated tangential relative
motion and last local TOI. The update returns a new immutable trial state:

- accepting the substep commits the returned state;
- rejecting it retains the original state unchanged;
- newly detected pairs appear with zero prior history;
- pairs absent from the next accepted step disappear.

This establishes dynamic pair lifetime and rollback semantics. The stored
tangential history is not yet used by a frictional impact law.

## Verification contracts

1. A vertex moves from `z=1` to `z=-1` through a zero-thickness triangle in a
   single step. Computed TOI is `0.49999999995` versus exact `0.5`; the accepted
   point lies on the face and cannot tunnel.
2. A triangle moving upward by `0.2` while the vertex moves downward by `2`
   gives independent exact TOI `1/2.2`. Relative TOI error is below `1e-8`.
3. One full step and the split sequence `0 -> 0.4 -> 1.0` produce the same
   global TOI to below `1e-8`.
4. Two moving triangular flaps collide at mid-step. Assembled impulse and
   moment residuals are below the 3% gate and the automated absolute limits
   are respectively `1e-11` and `1e-9`.
5. Separation removes the dynamic pair. Invalid masses and degenerate endpoint
   triangles fail closed.

The high-speed analytical TOI relative error is approximately `1e-10`, far
below 3%; force and moment residuals in the canonical case are exactly zero in
double precision.

```bash
PYTHONWARNINGS=error PYTHONPATH=src python -m pytest -q \
  tests/test_self_contact_ccd.py
```
