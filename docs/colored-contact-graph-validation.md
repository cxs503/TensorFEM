# Deterministically colored contact graphs

## Algorithm and data layout

`colored_contact_graph.py` greedily colors the sorted rigid-contact conflict
graph. Two constraints conflict when they share a dynamic rigid body; fixed
ground is not considered a shared body. Therefore all constraints of one color
can evaluate Delassus effective mass and update linear/angular velocities as a
Torch tensor batch without write conflicts.

The structure-of-arrays layout stores body indices, contact points, normals,
colors and accumulated impulses as tensors on one device. Kernels use
`index_add`, batched cross products and `einsum`; the same path is suitable for
CPU or CUDA tensors. CI qualification uses CPU when CUDA is unavailable.

Colors execute sequentially and constraints within a color execute
simultaneously. This color-Jacobi/PGS path does not follow the same intermediate
iteration sequence as scalar key-ordered PGS. For positive masses and positive
definite inertia, repeated color sweeps converge to the same frictionless
complementarity solution in the qualified cases.

## Verification

- A ten-body chain receives exactly two deterministic colors. No color shares
  a dynamic body, and reversing input constraint order produces identical
  color index tensors.
- An eight-body coupled chain matches scalar rigid PGS velocities within
  `1e-8`; momentum error is below `1e-10`, complementarity below `1e-8`, and
  kinetic energy is nonincreasing.
- Four independent 16-body chains form four islands and converge together with
  complementarity below `1e-7`.
- The performance regression has 1024 ground contacts. They form one color and
  1024 independent islands. On the current reference CPU a representative run
  took about `0.006 s` for the colored tensor path versus `1.125 s` for scalar
  PGS (roughly 187x); timing is diagnostic, while both paths have a 10 s
  fail-safe ceiling and must return identical velocities.
- Warm start reaches the cold solution with no more sweeps. Tensor device
  placement is checked for indices, geometry and color batches.

All comparison, conservation and complementarity errors are below 3%.

## Limits

This first colored kernel is frictionless. Frictional 2-D tangent blocks and
simultaneous-impact coupling remain on the scalar rigid path. Highly coupled
graphs may require many color sweeps; no universal convergence rate is
claimed. Actual multi-GPU scheduling, graph-color workers and asynchronous PGS
are not implemented, and ambiguous collinear CCD remains fail-closed.

```bash
PYTHONWARNINGS=error PYTHONPATH=src python -m pytest -q \
  tests/test_colored_contact_graph.py
```
