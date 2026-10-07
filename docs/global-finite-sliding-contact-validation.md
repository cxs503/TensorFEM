# Global two-dimensional finite-sliding contact

`global_contact2d.py` closes the gap between the local node/polyline contact
update and a finite-element equilibrium solve. Linear structural force,
external load and contact force share one global residual. The master polyline
is deformable: the closest segment is searched in the current configuration,
and the slave traction is distributed with the segment shape functions as
equal-and-opposite master forces.

Newton iterations always integrate friction from the immutable state committed
at the start of the increment. Only a converged trial state is retained. A
failed increment therefore leaves segment identity, elastic slip and cumulative
Coulomb dissipation unchanged. The global tangent differentiates the assembled
algorithmic residual in the selected active/stick/slip branch.

## Verification

The test model is a spring-supported slave node pressed onto a fixed straight
polyline and then dragged across its subdivision. Tests verify:

- free-DOF equilibrium and machine-precision equal/opposite contact forces;
- current-configuration candidate search while crossing a segment boundary;
- positive Coulomb dissipation and persistent slip history;
- zero spurious friction under common rigid translation of both sides;
- identical terminal response for one/twelve load increments and one/two
  collinear master segments;
- monotone penalty convergence to the rigid normal reaction, with less than
  3% error for the finest penalty;
- an algorithmic tangent against a centred residual difference; and
- explicit rollback after a deliberately rejected Newton increment.

Run with:

```bash
pytest -q tests/test_global_contact2d.py
PYTHONPATH=src python examples/global_finite_sliding_contact.py
```

## Capability boundary

This is a low-order 2-D node-to-deformable-polyline penalty formulation with
linear structural stiffness. It is not unbiased mortar contact, a general
continuum element assembly, self-contact, continuous collision detection, or
3-D surface contact. Segment switching is nonsmooth, so the tangent is valid
inside a fixed active/stick/slip branch and line search is used at transitions.
