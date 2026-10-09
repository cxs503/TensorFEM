# Shell4 arc-length real-discretization evidence

This evidence closes a narrow but important integration gap: the Crisfield
continuation solver now has a reproducible path test driven by an assembled
Shell4 residual and consistent tangent.  The model is a `1.0 x 0.2` flat
cantilever, represented by one four-node corotational facet.  Its left edge is
clamped and equal transverse nodal forces act on its right edge.

Every corrector evaluation calls `assemble_general_shell`.  Element internal
force is the gradient of the objective corotational energy and tangent is its
autograd Hessian.  The constrained global vectors and matrices are then passed
unchanged to the generic spherical arc-length solver.  Tests independently
reassemble every accepted point and require equilibrium residual divided by
the reference-load norm below `1e-7`.

Two paths traverse the same total arc length (`0.4`): four steps of `0.1` and
eight steps of `0.05`.  Their endpoint load factors and tip displacements must
agree within 3%.  The fine path reaches about `-0.126` transverse displacement,
large enough to exercise configuration updates rather than only the zero-state
adapter.

Run the evidence with:

```bash
PYTHONPATH=src python examples/marine/shell4_arc_path.py
PYTHONPATH=src python -m pytest -q tests/test_shell4_arc_path.py
```

## Qualification boundary

This is real finite-element residual/tangent and continuation evidence, but it
is deliberately **not** a general shell postbuckling qualification.  One flat
element does not establish spatial convergence, bifurcation branch switching,
imperfection sensitivity, follower loading, material plasticity, contact, or a
published shell snap-through benchmark.  Those claims remain gated on their
own multi-element reference cases.
