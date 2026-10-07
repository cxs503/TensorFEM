# Low-order mortar and self-contact foundations

## Mortar scope

`mortar_contact3d.py` integrates frictionless penalty traction at three-point
TRI3 or 2x2 QUAD4 slave quadrature points. Slave forces use slave shape
functions; each integration point is projected onto the current master mesh
and the opposite force uses master barycentric weights. This differs from the
older node-tributary-area scheme: geometry, traction and both surface force
vectors are coupled at actual surface integration points.

This first implementation is small-sliding, frictionless penalty coupling. It
does not claim a dual Lagrange basis, segment clipping, unbiased two-pass
mortar, finite-sliding history, frictional mortar or a global contact Newton
solver.

## Verification

1. A unit constant-pressure patch with a 3x3 slave mesh against a nonmatching
   2x2 master mesh reproduces the analytical resultant. Slave/master force and
   moment sums vanish to round-off, verifying consistent nodal forces.
2. Exchanging master and slave, with the receiving surface orientation
   reversed, changes the constant-patch resultant by less than `1e-10`
   relative. This measures the limited patch case only; it is not a claim of
   general unbiased mortar symmetry.
3. A curved slave surface against a nonmatching planar master has the following
   errors against independent high-resolution quadrature:

| slave subdivisions | resultant error |
|---:|---:|
| 1 | 10.00001% |
| 2 | 2.50001% |
| 4 | 0.62501% |
| 8 | 0.15626% |

The single-cell mesh is explicit failing convergence evidence. The formal
4-cell result is below the 3% gate.

## Self-contact foundation

`self_contact_candidates` builds conservative axis-aligned-box candidate pairs
and excludes:

- the same facet;
- all facet pairs sharing a vertex or edge;
- degenerate triangles and collapsed QUAD4 splits.

A connected four-triangle strip is folded back until its first and fourth
faces are within the search radius. The nonadjacent pair is detected, while
the three topologically adjacent pairs are absent. This qualifies candidate
filtering only. Contact enforcement, continuous collision detection,
two-sided normals and history for a dynamically folding body remain future
work; the candidate routine alone is not advertised as a self-contact solver.

```bash
PYTHONWARNINGS=error PYTHONPATH=src python -m pytest -q \
  tests/test_mortar_contact3d.py
```
